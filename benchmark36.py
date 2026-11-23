# -*- coding: utf-8 -*-
"""
benchmark36.py — comparação SEM vazamento de rótulos nos 36 datasets do artigo do
Supervised K-ISOMAP (Mendes & Levada, Pattern Recognition 183, 115087), com as mesmas
métricas (acurácia balanceada e F1 macro, médias sobre 8 classificadores).

Protocolo (idêntico ao da Seção 3.5 daquele artigo):
  * 10 divisões estratificadas 50/50 (random_state = 42..51), d = 2;
  * TODO método (grafo, tangentes, rótulos, normalização opcional) é ajustado só no treino;
  * o teste é projetado por um mapeador fora da amostra sem rótulos e só então classificado;
  * rótulos de teste só entram nas métricas finais.

Uso:
    python benchmark36.py --run                 # roda tudo (retoma de onde parou)
    python benchmark36.py --run --quick         # 3 datasets x 2 divisões (teste rápido)
    python benchmark36.py --summary             # tabelas + Friedman + Wilcoxon (Holm)
    python benchmark36.py --selftest            # teste com datasets do sklearn (sem internet)
Saídas: bench36_runs.csv, bench36_summary.csv, bench36_stats.csv  (em SEI_OUT_DIR ou ao lado do script)
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score

import sei_isomap as S

try:
    import umap as _umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
try:
    import functools, sys as _sys
    from metric_learn import LMNN, LFDA
    # metric-learn 0.7 chama check_X_y/check_array com force_all_finite=..., argumento que o
    # scikit-learn >= 1.6 renomeou para ensure_all_finite. Shim de compatibilidade:
    def _compat(fn):
        @functools.wraps(fn)
        def w(*a, **kw):
            if "force_all_finite" in kw:
                kw["ensure_all_finite"] = kw.pop("force_all_finite")
            return fn(*a, **kw)
        return w
    for _n, _m in list(_sys.modules.items()):
        if _n.startswith("metric_learn"):
            for _f in ("check_X_y", "check_array"):
                if hasattr(_m, _f):
                    try:
                        setattr(_m, _f, _compat(getattr(_m, _f)))
                    except Exception:
                        pass
    HAS_ML = True
except Exception:
    HAS_ML = False

OUT_DIR = os.environ.get("SEI_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(OUT_DIR, "bench36_runs.csv")
SUMMARY = os.path.join(OUT_DIR, "bench36_summary.csv")
STATS = os.path.join(OUT_DIR, "bench36_stats.csv")

SPLIT_SEEDS = list(range(42, 52))
DIM = 2
MAX_N = 2000          # teto de amostras por dataset (custo O(n^3) dos métodos de difusão/geodésica)

# (rótulo na tabela, nome no OpenML, fração de amostras, nº de PCs ou None)  — Tabela 1 do K-ISOMAP
BENCH36 = [
    ("digits", "digits", 1.0, None), ("optdigits", "optdigits", 1.0, None),
    ("car-evaluation", "car-evaluation", 1.0, None), ("wine-quality-white", "wine-quality-white", 1.0, None),
    ("wine-quality-red", "wine-quality-red", 1.0, None), ("glass", "glass", 1.0, None),
    ("ecoli", "ecoli", 1.0, None), ("vowel", "vowel", 1.0, None), ("collins", "collins", 1.0, None),
    ("energy-efficiency", "energy-efficiency", 1.0, None), ("satimage", "satimage", 1.0, None),
    ("led24", "led24", 1.0, None), ("tic-tac-toe", "tic-tac-toe", 1.0, None),
    ("balance-scale", "balance-scale", 1.0, None), ("diabetes", "diabetes", 1.0, None),
    ("mfeat-karhunen", "mfeat-karhunen", 1.0, None), ("mfeat-pixel", "mfeat-pixel", 1.0, None),
    ("grub-damage", "grub-damage", 1.0, None), ("banknote-authentication", "banknote-authentication", 1.0, None),
    ("vehicle", "vehicle", 1.0, None), ("ionosphere", "ionosphere", 1.0, None),
    ("wall-robot-navigation", "wall-robot-navigation", 1.0, None),
    ("CIFAR_10_small", "CIFAR_10_small", 0.20, 30), ("pendigits", "pendigits", 0.20, None),
    ("artificial-characters", "artificial-characters", 0.25, None), ("waveform-5000", "waveform-5000", 1.0, None),
    ("nursery", "nursery", 0.25, None), ("eye_movements", "eye_movements", 0.30, None),
    ("Indian_pines", "Indian_pines", 0.25, None), ("GesturePhaseSegmentation", "GesturePhaseSegmentation", 0.25, None),
    ("zoo", "zoo", 1.0, None), ("thyroid-dis", "thyroid-dis", 1.0, None),
    ("one-hundred-plants-shape", "one-hundred-plants-shape", 1.0, None),
    ("one-hundred-plants-texture", "one-hundred-plants-texture", 1.0, None),
    ("MNIST_784", "mnist_784", 0.05, 50), ("cnae-9", "cnae-9", 1.0, None),
]


# ----------------------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------------------
def load_bench_dataset(label, oml_name, frac, n_pc, max_n=MAX_N):
    """Carrega do OpenML, codifica, (opcional) PCA não supervisionado, sub-amostra estratificada.
    A padronização e o PCA usam só X (sem rótulos), como no artigo do K-ISOMAP."""
    from sklearn.datasets import fetch_openml
    if label == "digits":      # 'digits' do artigo = load_digits do scikit-learn (1797 x 64, 10 classes)
        from sklearn.datasets import load_digits
        Xd, yd = load_digits(return_X_y=True)
        X = StandardScaler().fit_transform(Xd)
        return X, yd, {"n_original": int(X.shape[0]), "n_used": int(X.shape[0])}
    ds = None
    for ver in ("active", 1, 2, 3):
        try:
            ds = fetch_openml(name=oml_name, version=ver, as_frame=True)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(f"OpenML: dataset '{oml_name}' não encontrado (informe o data_id)")
    X = S.prep_openml_X(ds.data)
    y = LabelEncoder().fit_transform(np.array(ds.target).ravel().astype(str))
    n0 = X.shape[0]
    if frac < 1.0:
        X, y = _strat_take(X, y, frac)
    if n_pc is not None:
        X = PCA(n_components=min(n_pc, X.shape[1]), random_state=0).fit_transform(X)
    if X.shape[0] > max_n:
        X, y = S.stratified_subsample(X, y, max_n=max_n, random_state=0)
    # classes com <2 amostras não podem ser estratificadas
    cls, cnt = np.unique(y, return_counts=True)
    keep = np.isin(y, cls[cnt >= 2])
    X, y = X[keep], LabelEncoder().fit_transform(y[keep])
    return X, y, {"n_original": int(n0), "n_used": int(X.shape[0])}


def _strat_take(X, y, frac, seed=0):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]
        idx.append(rng.choice(ic, size=max(2, int(round(len(ic) * frac))), replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]


# ----------------------------------------------------------------------------------------
# Supervised K-ISOMAP (reimplementação a partir do Algoritmo 1 do artigo)
# ----------------------------------------------------------------------------------------
def kisomap_fit(Xtr, ytr, k, d, sign_align=False, eps=1e-6):
    """Retorna Y_train (N x d). Fiel ao Alg. 1: espaços tangentes por PCA local do patch
    {x_i} U N_k(x_i); curvatura K_ij^(l) = ||u_il - u_jl||; peso = min_l K (mesma classe) ou
    soma_l K (classes distintas); caminhos mínimos; MDS clássico.
    Diferenças documentadas: usamos os r = min(m, k+1) vetores singulares do patch (os demais
    autovetores da covariância têm autovalor 0 e são arbitrários) e Dijkstra em vez de
    Floyd-Warshall (mesmo resultado). sign_align=True alinha o sinal dos autovetores (o sinal
    do PCA é arbitrário), variante NÃO faz parte do artigo original."""
    N, m = Xtr.shape
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Xtr)
    _, idx = nbrs.kneighbors(Xtr)
    r = min(m, k + 1)
    U = np.zeros((N, r, m))
    for i in range(N):
        P = Xtr[idx[i]]
        P = P - P.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(P, full_matrices=False)
        U[i, :Vt.shape[0]] = Vt[:r]
    rows, cols, vals = [], [], []
    seen = set()
    for i in range(N):
        for j in idx[i, 1:]:
            j = int(j)
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            Ui, Uj = U[a], U[b]
            if sign_align:
                sgn = np.sign(np.sum(Ui * Uj, axis=1, keepdims=True))
                sgn[sgn == 0] = 1.0
                Uj = Uj * sgn
            Kv = np.linalg.norm(Ui - Uj, axis=1)          # K^(l), l = 1..r
            w = Kv.min() if ytr[a] == ytr[b] else Kv.sum()
            rows.append(a); cols.append(b); vals.append(max(float(w), eps))
    G = csr_matrix((vals, (rows, cols)), shape=(N, N))
    D = shortest_path(G, method="D", directed=False)
    if np.isinf(D).any():
        D[np.isinf(D)] = 2.0 * D[np.isfinite(D)].max()
    D2 = D ** 2
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J
    ev, V = np.linalg.eigh(B)
    o = np.argsort(ev)[::-1][:d]
    lam = np.clip(ev[o], 0, None)
    return V[:, o] * np.sqrt(lam)[None, :]


def kisomap_official_fit(Xtr, ytr, k, d, max_m=150):
    """Supervised K-ISOMAP fiel ao código OFICIAL (github.com/alexandrelevada/SK-ISOMAP, sup_kiso.py):
    grafo k-NN direcionado SEM o próprio ponto; autovetores da covariância m x m dos k vizinhos,
    ordenados do maior para o menor autovalor; delta = ||U_i - U_j|| por coluna (TODAS as m);
    peso = min(delta) se mesma classe, sum(delta) se classes distintas; peso 0 = sem aresta
    (como no networkx); caminhos mínimos; MDS clássico. Usa eigh no lugar de eig (covariância
    simétrica; difere só em sinais/base do subespaço nulo, que são arbitrários nos dois)."""
    N, m = Xtr.shape
    if m > max_m:
        raise ValueError(f"K-ISOMAP oficial guarda N x m x m autovetores; m={m} > {max_m}")
    nb = NearestNeighbors(n_neighbors=k + 1).fit(Xtr)
    _, idx = nb.kneighbors(Xtr)
    idx = idx[:, 1:]                                   # sem o próprio ponto
    U = np.zeros((N, m, m))
    for i in range(N):
        C = np.cov(Xtr[idx[i]].T)
        w, V = np.linalg.eigh(np.atleast_2d(C))
        U[i] = V[:, np.argsort(w)[::-1]]
    rows, cols, vals = [], [], []
    for i in range(N):
        for j in idx[i]:
            delta = np.linalg.norm(U[i] - U[j], axis=0)
            w_ij = delta.min() if ytr[i] == ytr[j] else delta.sum()
            if w_ij > 0:
                rows.append(i); cols.append(int(j)); vals.append(float(w_ij))
    G = csr_matrix((vals, (rows, cols)), shape=(N, N))
    D = shortest_path(G, method="D", directed=False)
    if np.isinf(D).any():
        D[np.isinf(D)] = 2.0 * D[np.isfinite(D)].max()
    J = np.eye(N) - np.ones((N, N)) / N
    ev, V = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    o = np.argsort(ev)[::-1][:d]
    return V[:, o] * np.sqrt(np.clip(ev[o], 0, None))[None, :]


def rbf_extend(Xtr, Ytr, Xte):
    """Mapeador externo do artigo do K-ISOMAP: RBF linear, suavização 1e-8, grau 0, <=50 vizinhos."""
    nb = min(50, Xtr.shape[0])
    f = RBFInterpolator(Xtr, Ytr, kernel="linear", smoothing=1e-8, degree=0, neighbors=nb)
    return f(Xte)


# ----------------------------------------------------------------------------------------
# Métodos: cada um devolve (F_treino, F_teste)
# ----------------------------------------------------------------------------------------
def _sei(Xtr, ytr, Xte, d, k, eta_mult=1.0):
    A, _, _, _ = S.build_knn_graph_and_scale(Xtr, ytr, n_neighbors=k, eta_same=S.ETA_SAME * eta_mult,
                                             eta_diff=S.ETA_DIFF * eta_mult)
    Dd = S.diffusion_distance_from_affinity(A, steps=S.DIFF_STEPS)
    Ytr, Zte, _ = S._sei_nystrom_extension(A, Dd, Xtr, Xte, k, d)
    return Ytr, Zte


def embed(method, Xtr, ytr, Xte, d, seed):
    k = max(3, int(round(np.sqrt(Xtr.shape[0]))))
    k = min(k, Xtr.shape[0] - 2)
    if method == "RAW":
        return Xtr, Xte
    if method == "PCA":
        p = PCA(n_components=min(d, Xtr.shape[1]), random_state=seed).fit(Xtr)
        return p.transform(Xtr), p.transform(Xte)
    if method == "LDA":
        nc = min(d, len(np.unique(ytr)) - 1, Xtr.shape[1])
        l = LinearDiscriminantAnalysis(n_components=max(1, nc)).fit(Xtr, ytr)
        return l.transform(Xtr), l.transform(Xte)
    if method == "NCA":
        n = NeighborhoodComponentsAnalysis(n_components=d, random_state=seed).fit(Xtr, ytr)
        return n.transform(Xtr), n.transform(Xte)
    if method == "LMNN":
        l = LMNN(n_neighbors=3, n_components=d, max_iter=200, random_state=seed).fit(Xtr, ytr)
        return l.transform(Xtr), l.transform(Xte)
    if method == "LFDA":
        # LFDA resolve um autoproblema generalizado que fica singular com colunas colineares/
        # constantes (ex.: one-hot). PCA prévia (só em X) até o posto numérico resolve.
        p = PCA(n_components=min(30, Xtr.shape[1], Xtr.shape[0] - 1), random_state=seed).fit(Xtr)
        keep = int(np.sum(p.explained_variance_ > 1e-8 * p.explained_variance_[0]))
        p = PCA(n_components=max(d, keep), random_state=seed).fit(Xtr) if keep >= d else p
        Ptr, Pte = p.transform(Xtr), p.transform(Xte)
        l = LFDA(n_components=d, k=min(7, Xtr.shape[0] - 1)).fit(Ptr, ytr)
        return l.transform(Ptr), l.transform(Pte)
    if method == "SUPERVISED-UMAP":
        u = _umap.UMAP(n_components=d, n_neighbors=min(15, Xtr.shape[0] - 1), random_state=seed).fit(Xtr, ytr)
        return u.embedding_, u.transform(Xte)
    if method == "ISOMAP":
        iso = Isomap(n_neighbors=k, n_components=d, eigen_solver="dense").fit(Xtr)
        return iso.embedding_, iso.transform(Xte)
    if method == "ISOMAP+RBF":
        iso = Isomap(n_neighbors=k, n_components=d, eigen_solver="dense").fit(Xtr)
        return iso.embedding_, rbf_extend(Xtr, iso.embedding_, Xte)
    if method == "K-ISOMAP-OFF+RBF":
        Y = kisomap_official_fit(Xtr, ytr, k, d)
        return Y, rbf_extend(Xtr, Y, Xte)
    if method == "K-ISOMAP+RBF":
        Y = kisomap_fit(Xtr, ytr, k, d)
        return Y, rbf_extend(Xtr, Y, Xte)
    if method in ("SEI-NCA", "SEI0-NCA", "ISOMAP-NCA", "PCA-NCA", "SEI-NCA-S", "SEI-S"):
        Ftr, Fte, _ = S._fold_embed(method, Xtr, ytr, Xte, d, k, seed)   # fonte única: sei_isomap.py
        return Ftr, Fte
    if method == "SEI-ISOMAP":
        return _sei(Xtr, ytr, Xte, d, k)
    if method == "SEI-ISOMAP+RBF":
        Y, _ = _sei(Xtr, ytr, Xte, d, k)
        return Y, rbf_extend(Xtr, Y, Xte)
    raise ValueError(method)


def default_methods():
    m = ["SEI-NCA", "SEI-ISOMAP", "SEI-ISOMAP+RBF", "K-ISOMAP+RBF", "SEI0-NCA", "ISOMAP-NCA", "PCA-NCA", "ISOMAP", "ISOMAP+RBF", "NCA", "PCA", "LDA", "RAW"]
    if HAS_ML:
        m += ["LMNN", "LFDA"]
    if HAS_UMAP:
        m += ["SUPERVISED-UMAP"]
    return m


# ----------------------------------------------------------------------------------------
# Classificadores e métricas (8 classificadores, como no artigo)
# ----------------------------------------------------------------------------------------
def classifiers(seed):
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NB": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "MLP": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=seed),
        "DT": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "GP": GaussianProcessClassifier(random_state=seed),
    }


def eval_embedding(Ftr, ytr, Fte, yte, seed):
    out = {}
    sc = StandardScaler().fit(Ftr)
    A, B = sc.transform(Ftr), sc.transform(Fte)
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise ValueError("embedding com NaN/inf")
    for nm, clf in classifiers(seed).items():
        try:
            Atr, ytr_c = A, ytr
            if nm == "GP" and A.shape[0] > 600:     # GP é O(n^3): treina em 600 pontos
                rng = np.random.default_rng(seed)
                sel = rng.choice(A.shape[0], 600, replace=False)
                Atr, ytr_c = A[sel], ytr[sel]
                if len(np.unique(ytr_c)) < 2:
                    raise ValueError("1 classe")
            pred = clf.fit(Atr, ytr_c).predict(B)
            out[f"bacc_{nm}"] = float(balanced_accuracy_score(yte, pred))
            out[f"f1_{nm}"] = float(f1_score(yte, pred, average="macro"))
        except Exception:
            out[f"bacc_{nm}"] = np.nan
            out[f"f1_{nm}"] = np.nan
    out["bacc_mean"] = float(np.nanmean([out[k] for k in out if k.startswith("bacc_")]))
    out["f1_mean"] = float(np.nanmean([out[k] for k in out if k.startswith("f1_")]))
    return out


# ----------------------------------------------------------------------------------------
# Execução com retomada
# ----------------------------------------------------------------------------------------
def _done_keys():
    if not os.path.exists(RUNS):
        return set()
    d = pd.read_csv(RUNS, usecols=["dataset", "split", "method", "status"])
    d = d[d.status == "OK"]
    return set(zip(d.dataset.astype(str), d.split.astype(int), d.method.astype(str)))


def _append(row):
    """Anexa uma linha SEMPRE alinhada ao cabeçalho do arquivo (linhas ERROR não têm as colunas
    de métricas; sem este alinhamento elas desalinhavam o CSV)."""
    df = pd.DataFrame([row])
    if os.path.exists(RUNS):
        cols = list(pd.read_csv(RUNS, nrows=0).columns)
        df = df.reindex(columns=cols)
        df.to_csv(RUNS, mode="a", header=False, index=False)
    else:
        df.to_csv(RUNS, mode="a", header=True, index=False)


def run_dataset(label, X, y, methods, seeds, done, extra=None):
    for sd in seeds:
        todo = [m for m in methods if (label, sd, m) not in done]
        if not todo:
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=sd)
        for m in todo:
            row = {"dataset": label, "split": sd, "method": m, "n_train": len(ytr), "n_test": len(yte),
                   "n_features": X.shape[1], "n_classes": len(np.unique(y)), **(extra or {})}
            t0 = time.time()
            try:
                Ftr, Fte = embed(m, Xtr, ytr, Xte, DIM, sd)
                row.update(eval_embedding(Ftr, ytr, Fte, yte, sd))
                row["status"], row["error"] = "OK", ""
            except Exception as e:
                row["status"], row["error"] = "ERROR", f"{type(e).__name__}: {str(e)[:120]}"
            row["seconds"] = round(time.time() - t0, 2)
            _append(row)
            print(f"[B36][{label}] split={sd} {m} -> {row['status']}"
                  + (f" bacc={row.get('bacc_mean', float('nan')):.3f}" if row["status"] == "OK" else f" {row['error']}"),
                  flush=True)


def run(quick=False, methods=None, max_n=MAX_N, only=None, n_splits=None):
    warnings.filterwarnings("ignore")
    methods = methods or default_methods()
    done = _done_keys()
    seeds = SPLIT_SEEDS[:2] if quick else SPLIT_SEEDS[:(n_splits or len(SPLIT_SEEDS))]
    todo = BENCH36[:3] if quick else BENCH36
    if only:
        todo = [t for t in BENCH36 if t[0] in only]
    print(f"[B36] métodos={methods}\n[B36] splits={seeds} datasets={len(todo)} max_n={max_n}")
    for label, name, frac, npc in todo:
        try:
            X, y, meta = load_bench_dataset(label, name, frac, npc, max_n=max_n)
        except Exception as e:
            print(f"[B36][PULOU] {label}: {type(e).__name__}: {e}", flush=True)
            continue
        run_dataset(label, X, y, methods, seeds, done, extra=meta)
    print("[B36] Concluído")


# ----------------------------------------------------------------------------------------
# Resumo e estatística
# ----------------------------------------------------------------------------------------
# Nomes de exibição (tabelas/artigo). O CSV e as chaves de retomada mantêm os nomes internos.
DISPLAY = {
    # Artigo 1 (SEI-ISOMAP): grafo supervisionado, gamma fixo, sem refinamento
    "SEI-ISOMAP": "SEI-ISOMAP",
    "SEI-ISOMAP+RBF": "SEI-ISOMAP + RBF",
    # Artigo 2 (SIEDP): refinamento discriminante (+ gamma/s na versão final)
    "SEI-NCA-S": "SIEDP",
    "SEI-NCA": "SIEDP (gamma fixo)",
    "SEI-S": "SIEDP sem refinamento",
    "SEI0-NCA": "SIEDP (eta=0)",
    "ISOMAP-NCA": "ISOMAP + refinamento",
    "PCA-NCA": "PCA + refinamento",
    "K-ISOMAP+RBF": "Supervised K-ISOMAP",
    "SUPERVISED-UMAP": "Supervised UMAP",
}


def summarize(metric="bacc_mean"):
    from scipy.stats import friedmanchisquare, wilcoxon
    d = pd.read_csv(RUNS)
    d = d[d.status == "OK"]
    out = []
    for met in ("bacc_mean", "f1_mean"):
        tab = d.groupby(["dataset", "method"])[met].mean().unstack().rename(columns=DISPLAY)
        tab.to_csv(SUMMARY.replace(".csv", f"_{met}.csv"))
        # compara só os datasets em que TODOS os métodos principais rodaram
        core = [c for c in tab.columns if c in DISPLAY.values() or c in ("NCA", "LDA", "RAW", "ISOMAP", "PCA")]
        tab = tab.loc[tab[[c for c in ("SIEDP (gamma fixo)", "Supervised K-ISOMAP", "NCA")
                           if c in tab.columns]].notna().all(axis=1)]
        full = tab.dropna(axis=1, how="any")        # métodos presentes em todos os datasets restantes
        ranks = full.rank(axis=1, ascending=False).mean().sort_values()
        print(f"\n=== {met}: {tab.shape[0]} datasets ===")
        print(pd.DataFrame({"media": full.mean(), "rank_medio": full.rank(axis=1, ascending=False).mean()})
              .sort_values("rank_medio").round(4))
        if full.shape[1] >= 3 and full.shape[0] >= 3:
            st, p = friedmanchisquare(*[full[c] for c in full.columns])
            print(f"Friedman: chi2={st:.2f} p={p:.4g}")
            out.append({"metric": met, "test": "friedman", "pair": "all", "stat": st, "p": p})
        for ref in [c for c in ("SIEDP", "SIEDP (gamma fixo)", "SEI-ISOMAP") if c in tab.columns]:
            rows = []
            for m in tab.columns:
                if m == ref:
                    continue
                ok = tab[ref].notna() & tab[m].notna()
                x, y = tab.loc[ok, ref], tab.loc[ok, m]
                if ok.sum() < 5 or (x == y).all():
                    continue
                p = wilcoxon(x, y).pvalue
                rows.append((m, int((x > y).sum()), int((x < y).sum()), float((x - y).mean()), p, int(ok.sum())))
            rows.sort(key=lambda r: r[4])
            mm = len(rows)
            for rank_i, (m, w, l, dm, p, n) in enumerate(rows):   # Holm
                padj = min(1.0, p * (mm - rank_i))
                out.append({"metric": met, "test": f"wilcoxon_{ref}_vs", "pair": m, "wins": w, "losses": l,
                            "mean_diff": dm, "p": p, "p_holm": padj, "n": n})
    pd.DataFrame(out).to_csv(STATS, index=False)
    print(f"\n[B36] salvo: {STATS}")


# ----------------------------------------------------------------------------------------
def selftest():
    from sklearn.datasets import load_digits, load_wine, load_iris, load_breast_cancer
    warnings.filterwarnings("ignore")
    global RUNS
    RUNS = os.path.join(OUT_DIR, "bench36_selftest_runs.csv")
    if os.path.exists(RUNS):
        os.remove(RUNS)
    done = set()
    for nm, f in (("iris", load_iris), ("wine", load_wine), ("bc", load_breast_cancer), ("digits", load_digits)):
        X, y = f(return_X_y=True)
        X, y = S.stratified_subsample(StandardScaler().fit_transform(X), y, max_n=600, random_state=0)
        run_dataset(nm, X, y, default_methods(), SPLIT_SEEDS[:2], done)
    d = pd.read_csv(RUNS)
    print(d[d.status == "OK"].groupby("method")[["bacc_mean", "f1_mean", "seconds"]].mean().round(3))
    print("ERROS:", d[d.status != "OK"].groupby("method").size().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--max-n", type=int, default=MAX_N)
    ap.add_argument("--splits", type=int, default=None, help="nº de divisões 50/50 (padrão 10)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.run:
        run(quick=a.quick, methods=a.methods, max_n=a.max_n, only=a.datasets, n_splits=a.splits)
    elif a.summary:
        summarize()
    else:
        ap.print_help()