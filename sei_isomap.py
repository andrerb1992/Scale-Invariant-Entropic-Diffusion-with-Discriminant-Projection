# -*- coding: utf-8 -*-
"""
SEI-ISOMAP Experiment Runner (resumível + salvando incrementalmente)

O que este script faz:
- Roda SEI-ISOMAP e baselines em múltiplas dimensões, seeds e níveis de ruído.
- Salva incrementalmente (a cada método) em results_runs.csv.
- Se você parar no meio, ao reexecutar ele retoma do ponto onde parou (skip do que já existe no CSV).
- Robusto: se um método falhar em uma combinação (dataset/seed/noise/dim), ele salva uma linha com status=ERROR
  e continua (e no resume ele não fica tentando a mesma combinação indefinidamente).
- Gera results_summary.csv, stats_wilcoxon.csv e dataset_reduction.csv (sempre que possível).

Saídas:
- OUT_DIR/results_runs.csv (incremental + resumível)
- OUT_DIR/results_summary.csv
- OUT_DIR/stats_wilcoxon.csv
- OUT_DIR/dataset_reduction.csv
- OUT_DIR/figures/<dataset>/*.png (plots 2D, só seed=primeira e noise=0.0)

Obs:
- UMAP entra se estiver instalado.
"""

import warnings
warnings.simplefilter(action="ignore")

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend não-interativo (sem Tkinter) — evita crash em execução background
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from sklearn.manifold import trustworthiness, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.metrics import silhouette_score, pairwise_distances, accuracy_score, f1_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, kneighbors_graph
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import NeighborhoodComponentsAnalysis

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, wilcoxon
from scipy.interpolate import RBFInterpolator
from statsmodels.stats.multitest import multipletests

HAS_UMAP = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
# =========================
# CONFIG GERAL
# =========================
LABEL_NOISE_RATES = [0.0, 0.1, 0.2, 0.3]
RANDOM_SEEDS = list(range(10))
N_SPLITS_CV = 5

# Dimensões para avaliação (2 também para plot)

EMBED_PLOT = 2
EMBED_DIMS = [2, 5, 10, 20]

# Grafo / Difusão
N_NEIGHBORS_BASE = 10
K_PER_CLASS = 3

DIFF_GAMMA = 1.0
DIFF_STEPS = 3

ETA_SAME = 0.05
ETA_DIFF = 0.20

# Supervisão localmente adaptativa: eta_ij = eta * (1 + kappa * rho_ij),
# rho_ij = média da impureza local (fração de vizinhos de outra classe) de i e j.
ADAPTIVE_SUPERVISION = False   # flag global, False = comportamento atual
ADAPTIVE_KAPPA = 1.0           # intensidade da adaptação local

# Métricas com amostragem
PAIR_SAMPLES = 8000
SIL_SAMPLE_MAX = 800

# Subamostragem (evita OOM)
MAX_SAMPLES_RUN = 600

# Plots
PLOT_MAX_POINTS = 5000
PLOT_DPI = 200

# SAÍDAS
# Diretório de saída: variável de ambiente SEI_OUT_DIR, ou a pasta deste script.
OUT_DIR = os.environ.get("SEI_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RUNS_PATH = os.path.join(OUT_DIR, "results_runs.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "results_summary.csv")
STATS_PATH = os.path.join(OUT_DIR, "stats_wilcoxon.csv")
REDUCTION_PATH = os.path.join(OUT_DIR, "dataset_reduction.csv")

# =========================
# CONFIG DA ABLAÇÃO (gamma, t, contraste supervisionado)
# =========================
# Roda em `python sei_isomap.py --ablation`. Usa dim=2, sem ruído de rótulo
# (já cobertos pelo pipeline principal), em 4 datasets representativos.
ABLATION_DIM = 2
ABLATION_NOISE = 0.0
ABLATION_DATASETS = {"Iris": 61, "Wine": 187, "Vehicle": 54, "Digits": 28}
ABLATION_GAMMA_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]
ABLATION_T_GRID = [1, 2, 3, 5, 10]
ABLATION_SUPERVISION_MULT_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]  # multiplica ETA_SAME/ETA_DIFF padrão

ABLATION_OUT_DIR = os.path.join(OUT_DIR, "ablation")
ABLATION_RUNS_PATH = os.path.join(ABLATION_OUT_DIR, "ablation_results.csv")
ABLATION_SUMMARY_PATH = os.path.join(ABLATION_OUT_DIR, "ablation_summary.csv")

# Comparação da supervisão adaptativa (`python sei_isomap.py --adaptive-compare`).
# kappa=0.0 = baseline com ADAPTIVE_SUPERVISION=False.
ADAPTIVE_KAPPA_GRID = [0.5, 1.0, 2.0]
ADAPTIVE_RUNS_PATH = os.path.join(ABLATION_OUT_DIR, "adaptive_comparison_runs.csv")
ADAPTIVE_SUMMARY_PATH = os.path.join(ABLATION_OUT_DIR, "adaptive_comparison.csv")

# Avaliação indutiva sem vazamento (`python sei_isomap.py --inductive-eval`).
# Split 50/50 estratificado; embedding só no treino; mapeador RBF (agnóstico a
# rótulo) leva X_test -> Z_test. Imputação/padronização ajustadas só no treino.
INDUCTIVE_DIM = 2
INDUCTIVE_NOISE = 0.0
INDUCTIVE_TEST_SIZE = 0.5
INDUCTIVE_EXCLUDED = {"machine_cpu_v1", "newton_hema_v1", "Servo_ID"}
INDUCTIVE_RUNS_PATH = os.path.join(ABLATION_OUT_DIR, "inductive_eval_runs.csv")
INDUCTIVE_SUMMARY_PATH = os.path.join(ABLATION_OUT_DIR, "inductive_eval_summary.csv")
INDUCTIVE_WILCOXON_PATH = os.path.join(ABLATION_OUT_DIR, "inductive_eval_wilcoxon.txt")


# =========================
# UMAP (opcional)
# =========================
HAS_UMAP = False
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# =========================
# UTILIDADES
# =========================
def safe_dirname(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", str(name)).strip()

def stratified_subsample(X, y, max_n=3000, random_state=0):
    n = X.shape[0]
    if n <= max_n:
        return X, y

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    frac = max_n / n

    idx_all = []
    for c, cnt in zip(classes, counts):
        idx_c = np.where(y == c)[0]
        take = max(1, int(np.round(cnt * frac)))
        take = min(take, len(idx_c))
        idx_all.append(rng.choice(idx_c, size=take, replace=False))

    idx = np.unique(np.concatenate(idx_all))
    if len(idx) > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)

    return X[idx], y[idx]

def prep_openml_X(df_X: pd.DataFrame) -> np.ndarray:
    Xo = pd.get_dummies(df_X, dummy_na=True)
    X = SimpleImputer(strategy="median").fit_transform(Xo)
    X = StandardScaler().fit_transform(X)
    return X.astype(np.float64)

def add_label_noise(y, noise_rate, seed):
    rng = np.random.default_rng(seed)
    y = y.copy()
    n = len(y)
    m = int(np.round(noise_rate * n))
    if m == 0:
        return y
    classes = np.unique(y)
    if len(classes) < 2:
        return y
    idx = rng.choice(n, size=m, replace=False)
    for i in idx:
        y[i] = rng.choice(classes[classes != y[i]])
    return y

def load_openml(data_id: int, subsample_max=MAX_SAMPLES_RUN):
    ds = fetch_openml(data_id=data_id, as_frame=True)
    X = prep_openml_X(ds.data)

    y_raw = np.array(ds.target).ravel()
    y = LabelEncoder().fit_transform(y_raw.astype(str))

    n_original = int(X.shape[0])
    X, y = stratified_subsample(X, y, max_n=subsample_max, random_state=0)
    n_used = int(X.shape[0])

    meta = {
        "openml_name": ds.details.get("name", str(data_id)),
        "openml_id": ds.details.get("id", data_id),
        "openml_version": ds.details.get("version", None),
        "n_original": n_original,
        "n_used": n_used,
        "n_removed": int(n_original - n_used),
        "reduction_pct": float(100.0 * (n_original - n_used) / max(1, n_original)),
        "subsample_max": int(subsample_max),
    }
    return X, y, meta

def load_openml_by_name(name: str, version=None, try_versions=30, subsample_max=MAX_SAMPLES_RUN):
    if version is not None:
        ds = fetch_openml(name=name, version=version, as_frame=True)
    else:
        try:
            ds = fetch_openml(name=name, version="active", as_frame=True)
        except Exception:
            last_err = None
            ds = None
            for v in range(1, try_versions + 1):
                try:
                    ds = fetch_openml(name=name, version=v, as_frame=True)
                    break
                except Exception as e:
                    last_err = e
            if ds is None:
                raise last_err

    X = prep_openml_X(ds.data)
    y_raw = np.array(ds.target).ravel()
    y = LabelEncoder().fit_transform(y_raw.astype(str))

    n_original = int(X.shape[0])
    X, y = stratified_subsample(X, y, max_n=subsample_max, random_state=0)
    n_used = int(X.shape[0])

    meta = {
        "openml_name": ds.details.get("name", name),
        "openml_id": ds.details.get("id", None),
        "openml_version": ds.details.get("version", None),
        "n_original": n_original,
        "n_used": n_used,
        "n_removed": int(n_original - n_used),
        "reduction_pct": float(100.0 * (n_original - n_used) / max(1, n_original)),
        "subsample_max": int(subsample_max),
    }
    return X, y, meta


# =========================
# IO: incremental + resume
# =========================
KEY_COLS = ["dataset", "seed", "label_noise", "dim", "method"]

def _coerce_key_tuple(dataset, seed, noise, dim, method):
    # padroniza noise em 1 casa decimal para evitar problemas de float
    try:
        noise = round(float(noise), 1)
    except Exception:
        pass
    return (str(dataset), int(seed), noise, int(dim), str(method))

def load_done_keys(runs_path: str):
    """Lê results_runs.csv existente e retorna set de chaves já concluídas (inclui OK e ERROR)."""
    if not os.path.exists(runs_path):
        return set()

    try:
        df = pd.read_csv(runs_path, usecols=KEY_COLS)
    except Exception:
        df = pd.read_csv(runs_path)

    for c in KEY_COLS:
        if c not in df.columns:
            return set()

    keys = set()
    for r in df[KEY_COLS].itertuples(index=False, name=None):
        keys.add(_coerce_key_tuple(*r))
    return keys

def append_row_csv(row: dict, path: str):
    """Append de uma linha no CSV (cria com header se não existir)."""
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header, encoding="utf-8")


# =========================
# MÉTRICAS (SEM NxN)
# =========================
def pairwise_corr_from_pairs(X, emb, sample_pairs=20000, random_state=0):
    rng = np.random.default_rng(random_state)
    N = X.shape[0]
    if N < 2:
        return np.nan

    i = rng.integers(0, N, size=sample_pairs)
    j = rng.integers(0, N, size=sample_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    if len(i) == 0:
        return np.nan

    dX = np.linalg.norm(X[i] - X[j], axis=1)
    dY = np.linalg.norm(emb[i] - emb[j], axis=1)

    r, _ = pearsonr(dX, dY)
    return float(r)

def compute_metrics(X, emb, y, seed, n_neighbors_trust):
    n = X.shape[0]
    ktrust = min(max(1, n_neighbors_trust), n - 1)

    tw = trustworthiness(X, emb, n_neighbors=ktrust)

    if len(np.unique(y)) < 2 or n < 3:
        sil = np.nan
    else:
        sample_size = min(SIL_SAMPLE_MAX, n - 1)
        sil = silhouette_score(emb, y, sample_size=sample_size, random_state=seed)

    corr = pairwise_corr_from_pairs(X, emb, sample_pairs=PAIR_SAMPLES, random_state=seed)

    return {
        "trustworthiness": float(tw),
        "distance_corr": float(corr),
        "silhouette": float(sil) if np.isfinite(sil) else np.nan,
    }

def eval_classifiers(emb, y, seed):
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=seed)

    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=6, weights="distance"))
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale"))
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)  # Windows-friendly

    out = {}
    out["KNN_acc"] = cross_val_score(knn, emb, y, cv=cv, n_jobs=-1, scoring="accuracy").mean()
    out["SVM_acc"] = cross_val_score(svm, emb, y, cv=cv, n_jobs=-1, scoring="accuracy").mean()
    out["RF_acc"]  = cross_val_score(rf,  emb, y, cv=cv, n_jobs=-1, scoring="accuracy").mean()

    out["KNN_f1"]  = cross_val_score(knn, emb, y, cv=cv, n_jobs=-1, scoring="f1_macro").mean()
    out["SVM_f1"]  = cross_val_score(svm, emb, y, cv=cv, n_jobs=-1, scoring="f1_macro").mean()
    out["RF_f1"]   = cross_val_score(rf,  emb, y, cv=cv, n_jobs=-1, scoring="f1_macro").mean()

    return {k: float(v) for k, v in out.items()}


# =========================
# PLOT
# =========================
def save_embedding_plot(emb, y, title, out_path, max_points=PLOT_MAX_POINTS, random_state=0):
    emb = np.asarray(emb)
    y = np.asarray(y)

    if emb.shape[1] != 2:
        return

    n = emb.shape[0]
    if n > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        embp = emb[idx]
        yp = y[idx]
    else:
        embp = emb
        yp = y

    plt.figure(figsize=(7, 6))
    plt.scatter(embp[:, 0], embp[:, 1], c=yp, s=8, alpha=0.85)
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

def safe_lle(X, n_neighbors=10, n_components=2, random_state=0):
    for method, solver in [("standard", "auto"), ("standard", "dense"), ("modified", "dense")]:
        try:
            emb = LocallyLinearEmbedding(
                n_neighbors=n_neighbors,
                n_components=n_components,
                method=method,
                eigen_solver=solver,
                random_state=random_state
            ).fit_transform(X)
            return emb, f"{method}/{solver}"
        except Exception:
            pass

    emb = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
    return emb, "fallback/PCA"


# =========================
# SEI / SDM
# =========================
def classical_mds_from_dist(D, n_components=2):
    D2 = D**2
    N = D.shape[0]
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1]
    vals = np.clip(vals[idx], 0, None)
    vecs = vecs[:, idx]
    L = np.diag(np.sqrt(vals[:n_components] + 1e-12))
    V = vecs[:, :n_components]
    return V @ L

def safe_isomap(X, n_neighbors, n_components):
    """Isomap com fallback para autovalores negativos.

    Quando n_components fica próximo da dimensionalidade original (ex.: dim=400
    com X de 400 features), a decomposição espectral interna do KernelPCA usado
    pelo Isomap encontra autovalores negativos "significativos" (erro numérico
    esperado nesse regime) e o sklearn levanta ValueError em vez de clipar.
    Nesse caso, refazemos o MDS clássico manualmente (mesmo grafo kNN + caminho
    geodésico do Isomap) usando classical_mds_from_dist, que já clipa os
    autovalores negativos a zero.
    """
    try:
        emb = Isomap(
            n_neighbors=n_neighbors,
            n_components=n_components,
            eigen_solver="dense",  # evita ARPACK
        ).fit_transform(X)
        return emb, ""
    except ValueError as e:
        if "negative eigenvalues" not in str(e):
            raise

        knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode="distance")
        D = shortest_path(knn_graph, method="D", directed=False)

        if not np.isfinite(D).all():
            finite = D[np.isfinite(D)]
            fill = float(finite.max() * 1.5) if finite.size else 0.0
            D[~np.isfinite(D)] = fill

        emb = classical_mds_from_dist(D, n_components=n_components)
        return emb, "isomap_mds_fallback"

def corr_distmatrix_vs_embedding(D, emb, sample_pairs=20000, random_state=0):
    rng = np.random.default_rng(random_state)
    N = emb.shape[0]
    i = rng.integers(0, N, size=sample_pairs)
    j = rng.integers(0, N, size=sample_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    d_ref = D[i, j]
    d_emb = np.linalg.norm(emb[i] - emb[j], axis=1)
    r, _ = pearsonr(d_ref, d_emb)
    return float(r)

def consensus_labels(y, idx_nb, self_weight=1.0):
    """Rótulos de consenso local: voto majoritário entre os k vizinhos (+ o próprio ponto com
    peso self_weight). Empate mantém o rótulo original. Usa SÓ rótulos de treino."""
    y = np.asarray(y)
    classes, yi = np.unique(y, return_inverse=True)
    N, k = idx_nb.shape
    votes = np.zeros((N, len(classes)))
    for c in range(len(classes)):
        votes[:, c] = (yi[idx_nb] == c).sum(axis=1)
    votes[np.arange(N), yi] += self_weight
    best = votes.argmax(axis=1)
    top = votes.max(axis=1)
    own = votes[np.arange(N), yi]
    out = np.where(own >= top, yi, best)
    return classes[out]


_LAST_COST_SHIFT = 0.0   # deslocamento global dos custos usado na última construção do grafo
_LAST_SCALE = 1.0        # escala s (mediana das distâncias aos vizinhos) da última construção do grafo

# Modo do parâmetro de difusão gamma:
#   "fixed" : gamma = DIFF_GAMMA (comportamento original; depende da escala dos dados)
#   "scale" : gamma = GAMMA_SCALE / s  => invariante a reescalonar X (os custos já são ∝ s)
GAMMA_MODE = "fixed"
GAMMA_SCALE = 4.0


def _gamma_eff():
    return (GAMMA_SCALE / max(_LAST_SCALE, 1e-12)) if GAMMA_MODE == "scale" else DIFF_GAMMA



def build_knn_graph_and_scale(X, y, n_neighbors, eta_same, eta_diff, adaptive=None, kappa=None, robust=False):
    if adaptive is None:
        adaptive = ADAPTIVE_SUPERVISION
    if kappa is None:
        kappa = ADAPTIVE_KAPPA

    N = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    dist, idx = nbrs.kneighbors(X)

    if robust:
        y = consensus_labels(y, idx[:, 1:])
    s = float(np.median(dist[:, 1:])) + 1e-12
    global _LAST_SCALE
    _LAST_SCALE = s
    alpha_same = -eta_same * s
    alpha_diff =  eta_diff * s

    if adaptive:
        # impureza local: fração dos k vizinhos de i (idx[:, 1:] = N_k(i)) com classe != y_i
        y_arr = np.asarray(y)
        impurity = (y_arr[idx[:, 1:]] != y_arr[:, None]).mean(axis=1)

    # Um único peso por aresta não-direcionada (regra OR-kNN): se i e j são
    # vizinhos um do outro, a aresta {i,j} é contada UMA vez (custo é simétrico).
    edges = {}
    for i in range(N):
        for k in range(1, n_neighbors + 1):
            j = int(idx[i, k])
            base = float(dist[i, k])

            if adaptive:
                rho_ij = 0.5 * (impurity[i] + impurity[j])  # simétrico, em [0,1]
                eta_same_ij = eta_same * (1.0 + kappa * rho_ij)
                eta_diff_ij = eta_diff * (1.0 + kappa * rho_ij)
                alpha_same_ij = -eta_same_ij * s
                alpha_diff_ij =  eta_diff_ij * s
                cost = base + (alpha_same_ij if y[i] == y[j] else alpha_diff_ij)
            else:
                cost = base + (alpha_same if y[i] == y[j] else alpha_diff)
            cost = float(max(cost, 0.0))
            edges[(min(i, j), max(i, j))] = cost

    keys = list(edges.keys())
    r = np.array([a for a, _ in keys], dtype=int)
    c = np.array([b for _, b in keys], dtype=int)
    # Estabilidade numérica: a = exp(-gamma*custo) UNDERFLOWA quando os custos são grandes
    # (ex.: 240 features padronizadas => distâncias ~20-40 => a ~ 1e-9..1e-18), e o antigo
    # "+1e-12" na normalização de linha passava a dominar a soma. Deslocar todos os custos
    # pelo mínimo global multiplica A por uma constante, o que NÃO altera P = D^-1 A nem a
    # matriz normalizada do SDM; matematicamente é o mesmo modelo.
    costs = np.array([edges[k] for k in keys], dtype=float)
    global _LAST_COST_SHIFT
    _LAST_COST_SHIFT = float(costs.min())
    v = np.exp(-_gamma_eff() * (costs - _LAST_COST_SHIFT))
    A = csr_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(N, N))
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr(), s, alpha_same, alpha_diff

def diffusion_distance_from_affinity(A, steps=3):
    row_sum = np.maximum(np.array(A.sum(axis=1)).ravel(), 1e-300)
    D_inv = 1.0 / row_sum
    P = A.multiply(D_inv[:, None])

    Pt = P.copy()
    for _ in range(steps - 1):
        Pt = Pt @ P

    Pt_dense = Pt.toarray().astype(np.float32)
    D = pairwise_distances(Pt_dense, metric="euclidean", n_jobs=-1)
    return D

def supervised_diffusion_maps(A, n_components, t_steps):
    d = np.maximum(np.array(A.sum(axis=1)).ravel(), 1e-300)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    S = A.multiply(d_inv_sqrt[:, None]).multiply(d_inv_sqrt[None, :]).tocsr()

    k = min(n_components + 1, S.shape[0] - 1)
    k = max(k, 2)

    try:
        vals, vecs = eigsh(S, k=k, which="LA", maxiter=20000, tol=1e-3)
        idx = np.argsort(vals)[::-1]
        vals = np.real(vals[idx])
        vecs = np.real(vecs[:, idx])
    except Exception:
        Sd = S.toarray()
        vals_all, vecs_all = np.linalg.eigh(Sd)
        idx = np.argsort(vals_all)[::-1][:k]
        vals = np.real(vals_all[idx])
        vecs = np.real(vecs_all[:, idx])

    vals = vals[1:n_components+1]
    vecs = vecs[:, 1:n_components+1]
    emb = vecs * (np.power(np.clip(vals, 0, None), t_steps)[None, :])
    return emb

def nca_embedding(X, y, n_components, seed):
    nca = NeighborhoodComponentsAnalysis(n_components=n_components, random_state=seed)
    return nca.fit_transform(X, y)


# =========================
# STATS (pós-processamento)
# =========================
def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    se = s / np.sqrt(len(x)) if len(x) > 1 else 0.0
    ci = 1.96 * se
    return m, s, m - ci, m + ci

def wilcoxon_stats(df_runs, dataset, noise, dim, metric,
                   base_method="SEI-ISOMAP", other_method="SDM"):
    a = df_runs[
        (df_runs["dataset"] == dataset) &
        (df_runs["label_noise"] == noise) &
        (df_runs["dim"] == dim) &
        (df_runs["method"] == base_method)
    ][metric].values

    b = df_runs[
        (df_runs["dataset"] == dataset) &
        (df_runs["label_noise"] == noise) &
        (df_runs["dim"] == dim) &
        (df_runs["method"] == other_method)
    ][metric].values

    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return np.nan

    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return np.nan

    if np.allclose(a, b):
        return 1.0

    try:
        stat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(stat.pvalue)
    except Exception:
        return np.nan

def postprocess_outputs(runs_path=None, summary_path=None, stats_path=None, write_reduction=True):
    """Gera summary e wilcoxon a partir do CSV de runs existente."""
    RUNS_PATH_ = runs_path or RUNS_PATH
    SUMMARY_PATH_ = summary_path or SUMMARY_PATH
    STATS_PATH_ = stats_path or STATS_PATH
    if not os.path.exists(RUNS_PATH_):
        print(f"[POST] {RUNS_PATH_} não existe ainda.")
        return

    df_runs = pd.read_csv(RUNS_PATH_)

    # usa só linhas OK para métricas e testes
    if "status" in df_runs.columns:
        df_ok = df_runs[df_runs["status"] == "OK"].copy()
    else:
        df_ok = df_runs.copy()

    metrics_cols = [
        "trustworthiness", "distance_corr", "silhouette",
        "KNN_acc", "SVM_acc", "RF_acc",
        "KNN_f1", "SVM_f1", "RF_f1",
        "sei_align_Ddiff"
    ]

    # Summary
    summary_rows = []
    for (dataset, noise, dim, method), g in df_ok.groupby(["dataset", "label_noise", "dim", "method"]):
        row = {"dataset": dataset, "label_noise": noise, "dim": dim, "method": method}
        for mc in metrics_cols:
            if mc in g.columns:
                m, s, lo, hi = mean_ci95(g[mc].values)
            else:
                m, s, lo, hi = (np.nan, np.nan, np.nan, np.nan)
            row[f"{mc}_mean"] = m
            row[f"{mc}_std"] = s
            row[f"{mc}_ci95_lo"] = lo
            row[f"{mc}_ci95_hi"] = hi
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).sort_values(
        ["dataset", "label_noise", "dim", "method"]
    ).reset_index(drop=True)
    df_summary.to_csv(SUMMARY_PATH_, index=False, encoding="utf-8")

    # Wilcoxon
    methods_to_compare = ["SDM", "ISOMAP", "PCA", "LAPLACIAN", "NCA"]
    if HAS_UMAP:
        methods_to_compare.append("UMAP")

    stats_rows = []
    for dataset in df_ok["dataset"].unique():
        for noise in sorted(df_ok[df_ok["dataset"] == dataset]["label_noise"].unique()):
            for dim in sorted(df_ok[(df_ok["dataset"] == dataset) & (df_ok["label_noise"] == noise)]["dim"].unique()):
                for other in methods_to_compare:
                    for metric in ["SVM_acc", "RF_acc", "KNN_acc", "SVM_f1", "RF_f1", "KNN_f1"]:
                        if metric not in df_ok.columns:
                            continue
                        p = wilcoxon_stats(df_ok, dataset, noise, dim, metric, "SEI-ISOMAP", other)
                        stats_rows.append({
                            "dataset": dataset,
                            "label_noise": noise,
                            "dim": dim,
                            "metric": metric,
                            "comparison": f"SEI-ISOMAP vs {other}",
                            "pvalue": p
                        })

    df_stats = pd.DataFrame(stats_rows).sort_values(
        ["dataset", "dim", "metric", "comparison"]
    ).reset_index(drop=True)

    # Correção de Holm-Bonferroni por família = (dataset, label_noise, dim, metric),
    # i.e. corrige dentro do grupo das comparações SEI-ISOMAP vs cada baseline
    # feitas naquela mesma condição experimental.
    df_stats["significant_raw"] = df_stats["pvalue"] < 0.05
    df_stats["pvalue_holm"] = np.nan
    df_stats["significant_holm"] = False
    for _, idx in df_stats.groupby(["dataset", "label_noise", "dim", "metric"]).groups.items():
        pvals = df_stats.loc[idx, "pvalue"].values
        mask_valid = ~pd.isna(pvals)
        if mask_valid.sum() == 0:
            continue
        reject, pvals_corr, _, _ = multipletests(pvals[mask_valid], alpha=0.05, method="holm")
        valid_idx = np.array(idx)[mask_valid]
        df_stats.loc[valid_idx, "pvalue_holm"] = pvals_corr
        df_stats.loc[valid_idx, "significant_holm"] = reject

    df_stats.to_csv(STATS_PATH_, index=False, encoding="utf-8")

    # Reduction (a partir do que existe no runs)
    red_cols = ["dataset", "openml_name", "openml_id", "openml_version",
                "subsample_max", "n_original", "n_used", "n_removed", "reduction_pct"]
    keep = [c for c in red_cols if c in df_runs.columns]
    if keep and write_reduction:
        df_red = df_runs[keep].drop_duplicates().sort_values(["dataset"]).reset_index(drop=True)
        df_red.to_csv(REDUCTION_PATH, index=False, encoding="utf-8")

    print("\n[POST] ✅ Atualizado:")
    print(f"- Summary:   {SUMMARY_PATH_}")
    print(f"- Stats:     {STATS_PATH_}")
    if write_reduction and os.path.exists(REDUCTION_PATH):
        print(f"- Reduction: {REDUCTION_PATH}")


# =========================
# EXECUÇÃO por dataset (resumível)
# =========================
def run_dataset_incremental(dataset_name, X, y, ds_meta, done_keys: set):
    dataset_name = safe_dirname(dataset_name)
    n = X.shape[0]
    if n < 10:
        print(f"[PULOU] {dataset_name}: N muito pequeno ({n})")
        return

    n_classes = len(np.unique(y))

    # k "cru" (pode explodir quando n_classes é enorme)
    k_graph_raw = min(max(N_NEIGHBORS_BASE, K_PER_CLASS * n_classes), n - 1)

    # cap para evitar grafos quase completos (também deixa mais estável)
    k_cap = max(2, (n // 2) - 1)      # mesma lógica de segurança
    k_graph = min(k_graph_raw, k_cap)
    k_trust = min(k_graph, n - 1)

    # Isomap precisa n_neighbors < n/2
    k_iso = min(k_graph, k_cap)

    p = X.shape[1]
    max_d = min(p, n - 1)

    dims = {d for d in EMBED_DIMS if d <= max_d}
    if max_d >= 2:
        dims.add(max_d)
    dims = sorted(dims)
    if len(dims) == 0:
        dims = [2] if max_d >= 2 else [1]

    ds_fig_dir = os.path.join(FIG_DIR, dataset_name)
    os.makedirs(ds_fig_dir, exist_ok=True)

    print(f"\n============================")
    print(f"Dataset: {dataset_name} | X={X.shape} | classes={n_classes}")
    print(f"k_graph_raw={k_graph_raw} | k_graph_used={k_graph} | k_iso={k_iso} | k_cap={k_cap}")
    print(f"Reduction: N {ds_meta['n_original']} -> {ds_meta['n_used']} ({ds_meta['reduction_pct']:.1f}%)")
    print(f"============================")

    first_seed = RANDOM_SEEDS[0]

    def base_row(method, seed, noise_key, d):
        return {
            "status": "OK",
            "error_msg": "",
            "dataset": dataset_name,
            "seed": int(seed),
            "label_noise": float(noise_key),
            "dim": int(d),
            "k_graph": int(k_graph),
            "k_iso": int(k_iso),
            "gamma": float(DIFF_GAMMA),
            "t_steps": int(DIFF_STEPS),
            "eta_same": float(ETA_SAME),
            "eta_diff": float(ETA_DIFF),
            "method": str(method),
            "openml_name": ds_meta["openml_name"],
            "openml_id": ds_meta["openml_id"],
            "openml_version": ds_meta["openml_version"],
            "subsample_max": ds_meta["subsample_max"],
            "n_original": ds_meta["n_original"],
            "n_used": ds_meta["n_used"],
            "n_removed": ds_meta["n_removed"],
            "reduction_pct": ds_meta["reduction_pct"],
        }

    def save_error(method, seed, noise_key, d, err):
        row = base_row(method, seed, noise_key, d)
        row["status"] = "ERROR"
        row["error_msg"] = f"{type(err).__name__}: {err}"
        # métricas vazias
        for col in ["trustworthiness","distance_corr","silhouette",
                    "KNN_acc","SVM_acc","RF_acc","KNN_f1","SVM_f1","RF_f1",
                    "sei_scale_s","alpha_same","alpha_diff","sei_align_Ddiff","lle_mode"]:
            row[col] = np.nan
        append_row_csv(row, RUNS_PATH)
        done_keys.add(_coerce_key_tuple(dataset_name, seed, noise_key, d, method))
        print(f"[WARN] {dataset_name} seed={seed} noise={noise_key:.1f} dim={d} {method} -> {row['error_msg']}")

    def missing(method, seed, noise_key, d):
        return _coerce_key_tuple(dataset_name, seed, noise_key, d, method) not in done_keys

    for seed in RANDOM_SEEDS:
        for noise in LABEL_NOISE_RATES:
            noise_key = round(float(noise), 1)
            y_noisy = add_label_noise(y, noise_key, seed)

            for d in dims:
                # já terminou tudo nessa combinação?
                method_list = ["SEI-ISOMAP", "SDM", "NCA", "PCA", "ISOMAP", "LLE", "LAPLACIAN"]
                if HAS_UMAP:
                    method_list.append("UMAP")
                if not any(missing(m, seed, noise_key, d) for m in method_list):
                    continue

                print(f"[{dataset_name}] seed={seed} noise={noise_key:.1f} dim={d}")

                # grafo supervisionado (só se precisar de SEI/SDM)
                A_sup = None
                scale_s = alpha_same = alpha_diff = None
                need_A = missing("SEI-ISOMAP", seed, noise_key, d) or missing("SDM", seed, noise_key, d)

                if need_A:
                    try:
                        A_sup, scale_s, alpha_same, alpha_diff = build_knn_graph_and_scale(
                            X, y_noisy, n_neighbors=k_graph, eta_same=ETA_SAME, eta_diff=ETA_DIFF
                        )
                    except Exception as e:
                        # se nem o grafo foi possível, SEI e SDM falham
                        if missing("SEI-ISOMAP", seed, noise_key, d):
                            save_error("SEI-ISOMAP", seed, noise_key, d, e)
                        if missing("SDM", seed, noise_key, d):
                            save_error("SDM", seed, noise_key, d, e)
                        # segue para métodos que não dependem do grafo
                        A_sup = None

                # --- SEI-ISOMAP
                if missing("SEI-ISOMAP", seed, noise_key, d):
                    try:
                        Ddiff = diffusion_distance_from_affinity(A_sup, steps=DIFF_STEPS)
                        emb = classical_mds_from_dist(Ddiff, n_components=d)
                        align = corr_distmatrix_vs_embedding(Ddiff, emb, sample_pairs=PAIR_SAMPLES, random_state=seed)

                        mets = compute_metrics(X, emb, y, seed=seed, n_neighbors_trust=k_trust)
                        cls = eval_classifiers(emb, y, seed=seed)

                        row = base_row("SEI-ISOMAP", seed, noise_key, d)
                        row.update(mets)
                        row.update(cls)
                        row["sei_scale_s"] = scale_s
                        row["alpha_same"] = alpha_same
                        row["alpha_diff"] = alpha_diff
                        row["sei_align_Ddiff"] = align
                        row["lle_mode"] = ""

                        append_row_csv(row, RUNS_PATH)
                        done_keys.add(_coerce_key_tuple(dataset_name, seed, noise_key, d, "SEI-ISOMAP"))

                        if d == EMBED_PLOT and seed == first_seed and noise_key == 0.0:
                            outp = os.path.join(ds_fig_dir, "SEI-ISOMAP.png")
                            if not os.path.exists(outp):
                                save_embedding_plot(emb, y, f"{dataset_name} - SEI-ISOMAP", outp, random_state=seed)

                    except Exception as e:
                        save_error("SEI-ISOMAP", seed, noise_key, d, e)

                # --- SDM
                if missing("SDM", seed, noise_key, d):
                    try:
                        emb = supervised_diffusion_maps(A_sup, n_components=d, t_steps=DIFF_STEPS)

                        mets = compute_metrics(X, emb, y, seed=seed, n_neighbors_trust=k_trust)
                        cls = eval_classifiers(emb, y, seed=seed)

                        row = base_row("SDM", seed, noise_key, d)
                        row.update(mets)
                        row.update(cls)
                        row["sei_scale_s"] = np.nan
                        row["alpha_same"] = np.nan
                        row["alpha_diff"] = np.nan
                        row["sei_align_Ddiff"] = np.nan
                        row["lle_mode"] = ""

                        append_row_csv(row, RUNS_PATH)
                        done_keys.add(_coerce_key_tuple(dataset_name, seed, noise_key, d, "SDM"))

                        if d == EMBED_PLOT and seed == first_seed and noise_key == 0.0:
                            outp = os.path.join(ds_fig_dir, "SDM.png")
                            if not os.path.exists(outp):
                                save_embedding_plot(emb, y, f"{dataset_name} - SDM", outp, random_state=seed)

                    except Exception as e:
                        save_error("SDM", seed, noise_key, d, e)

                # --- NCA (usa y_noisy)
                if missing("NCA", seed, noise_key, d):
                    try:
                        emb = nca_embedding(X, y_noisy, n_components=d, seed=seed)

                        mets = compute_metrics(X, emb, y, seed=seed, n_neighbors_trust=k_trust)
                        cls = eval_classifiers(emb, y, seed=seed)

                        row = base_row("NCA", seed, noise_key, d)
                        row.update(mets)
                        row.update(cls)
                        row["sei_scale_s"] = np.nan
                        row["alpha_same"] = np.nan
                        row["alpha_diff"] = np.nan
                        row["sei_align_Ddiff"] = np.nan
                        row["lle_mode"] = ""

                        append_row_csv(row, RUNS_PATH)
                        done_keys.add(_coerce_key_tuple(dataset_name, seed, noise_key, d, "NCA"))

                        if d == EMBED_PLOT and seed == first_seed and noise_key == 0.0:
                            outp = os.path.join(ds_fig_dir, "NCA.png")
                            if not os.path.exists(outp):
                                save_embedding_plot(emb, y, f"{dataset_name} - NCA", outp, random_state=seed)

                    except Exception as e:
                        save_error("NCA", seed, noise_key, d, e)

                # --- PCA
                if missing("PCA", seed, noise_key, d):
                    try:
                        emb = PCA(n_components=d, random_state=seed).fit_transform(X)

                        mets = compute_metrics(X, emb, y, seed=seed, n_neighbors_trust=k_trust)
                        cls = eval_classifiers(emb, y, seed=seed)

                        row = base_row("PCA", seed, noise_key, d)
                        row.update(mets)
                        row.update(cls)
                        row["sei_scale_s"] = np.nan
                        row["alpha_same"] = np.nan
                        row["alpha_diff"] = np.nan
                        row["sei_align_Ddiff"] = np.nan
                        row["lle_mode"] = ""

                        append_row_csv(row, RUNS_PATH)
                        done_keys.add(_coerce_key_tuple(dataset_name, seed, noise_key, d, "PCA"))

                        if d == EMBED_PLOT and seed == first_seed and noise_key == 0.0:
                            outp = os.path.join(ds_fig_dir, "PCA.png")
                            if not os.path.exists(outp):
                                save_embedding_plot(emb, y, f"{dataset_name} - PCA", outp, random_state=seed)

                    except Exception as e:
                        save_error("PCA", seed, noise_key, d, e)

                # --- ISOMAP (sem ARPACK)
                if missing("ISOMAP", seed, noise_key, d):
                    try:
                        emb, iso_mode = safe_isomap(X, n_neighbors=k_iso, n_components=d)

                        mets = compute_metrics(X, emb, y, seed=seed, n_neighbors_trust=k_trust)
                        cls = eval_classifiers(emb, y, seed=seed)

                        row = base_row("ISOMAP", seed, noise_key, d)
                        row.update(mets)
                        row.update(cls)
                        row["sei_scale_s"] = np.nan
                        row["alpha_same"] = np.nan
                        row["alpha_diff"] = np.nan
                        row["sei_align_Ddiff"] = np.nan
                        row["lle_mode"] = iso_mode

                        append_row_csv(row, RUNS_PATH)
                        done_keys.add(_coerce_key_tuple(dataset_name, seed, noise_key, d, "ISOMAP"))

                        if d == EMBED_PLOT and seed == first_seed and noise_key == 0.0:
                            outp = os.path.join(ds_fig_dir, "ISOMAP.png")
                            if not os.path.exists(outp):
                                save_embedding_plot(emb, y, f"{dataset_name} - ISOMAP", outp, random_state=seed)

                    except Exception as e:
                        save_error("ISOMAP", seed, noise_key, d, e)

                # --- LLE (nome unificado)
                if missing("LLE", seed, noise_key, d):
                    try:
                        k_lle = min(max(k_graph, d + 1), n - 1)
                        emb, lle_mode = safe_lle(X, n_neighbors=k_lle, n_components=d, random_state=seed)

                        mets = compute_metrics(X, emb, y, seed=seed, n_neighbors_trust=k_trust)
                        cls = eval_classifiers(emb, y, seed=seed)

                        row = base_row("LLE", seed, noise_key, d)
                        row.update(mets)
                        row.update(cls)
                        row["sei_scale_s"] = np.nan
                        row["alpha_same"] = np.nan
                        row["alpha_diff"] = np.nan
                        row["sei_align_Ddiff"] = np.nan
                        row["lle_mode"] = lle_mode

                        append_row_csv(row, RUNS_PATH)
                        done_keys.add(_coerce_key_tuple(dataset_name, seed, noise_key, d, "LLE"))

                        if d == EMBED_PLOT and seed == first_seed and noise_key == 0.0:
                            outp = os.path.join(ds_fig_dir, "LLE.png")
                            if not os.path.exists(outp):
                                save_embedding_plot(emb, y, f"{dataset_name} - LLE ({lle_mode})", outp, random_state=seed)

                    except Exception as e:
                        save_error("LLE", seed, noise_key, d, e)

                # --- LAPLACIAN / SpectralEmbedding
                if missing("LAPLACIAN", seed, noise_key, d):
                    try:
                        try:
                            emb = SpectralEmbedding(
                                n_components=d,
                                n_neighbors=k_graph,
                                random_state=seed
                            ).fit_transform(X)
                        except Exception:
                            emb = PCA(n_components=d, random_state=seed).fit_transform(X)

                        mets = compute_metrics(X, emb, y, seed=seed, n_neighbors_trust=k_trust)
                        cls = eval_classifiers(emb, y, seed=seed)

                        row = base_row("LAPLACIAN", seed, noise_key, d)
                        row.update(mets)
                        row.update(cls)
                        row["sei_scale_s"] = np.nan
                        row["alpha_same"] = np.nan
                        row["alpha_diff"] = np.nan
                        row["sei_align_Ddiff"] = np.nan
                        row["lle_mode"] = ""

                        append_row_csv(row, RUNS_PATH)
                        done_keys.add(_coerce_key_tuple(dataset_name, seed, noise_key, d, "LAPLACIAN"))

                        if d == EMBED_PLOT and seed == first_seed and noise_key == 0.0:
                            outp = os.path.join(ds_fig_dir, "LAPLACIAN.png")
                            if not os.path.exists(outp):
                                save_embedding_plot(emb, y, f"{dataset_name} - LAPLACIAN", outp, random_state=seed)

                    except Exception as e:
                        save_error("LAPLACIAN", seed, noise_key, d, e)

                # --- UMAP (se disponível)
                if HAS_UMAP and missing("UMAP", seed, noise_key, d):
                    try:
                        emb = umap.UMAP(
                            n_components=d,
                            n_neighbors=k_graph,
                            min_dist=0.1,
                            metric="euclidean",
                            random_state=seed
                        ).fit_transform(X)

                        mets = compute_metrics(X, emb, y, seed=seed, n_neighbors_trust=k_trust)
                        cls = eval_classifiers(emb, y, seed=seed)

                        row = base_row("UMAP", seed, noise_key, d)
                        row.update(mets)
                        row.update(cls)
                        row["sei_scale_s"] = np.nan
                        row["alpha_same"] = np.nan
                        row["alpha_diff"] = np.nan
                        row["sei_align_Ddiff"] = np.nan
                        row["lle_mode"] = ""

                        append_row_csv(row, RUNS_PATH)
                        done_keys.add(_coerce_key_tuple(dataset_name, seed, noise_key, d, "UMAP"))

                        if d == EMBED_PLOT and seed == first_seed and noise_key == 0.0:
                            outp = os.path.join(ds_fig_dir, "UMAP.png")
                            if not os.path.exists(outp):
                                save_embedding_plot(emb, y, f"{dataset_name} - UMAP", outp, random_state=seed)

                    except Exception as e:
                        save_error("UMAP", seed, noise_key, d, e)


# =========================
# DATASETS
# =========================
DATASET_IDS = {
    "Iris": 61,
    "Wine": 187,
    "Seeds": 1499,
    "Glass": 41,
    "Digits": 28,
    "Vehicle": 54,
    "Yeast": 181,
    "Blood": 1464,
    "KC1": 1067,
    "MFeatFourier": 14,
    "Cardio_ID": 1466,
    "Haberman_ID": 43,
    "HeartStatlog_ID": 53,
    "Sonar_ID": 40,
    "Servo_ID": 46292,
    "Speech_ID": 40910,
    "Datatrieve_ID": 40806,
    "HayesRoth_ID": 329,
    "MNIST": 554,
    "satimage_v1": 182,
}

DATASET_SPECS = [
    ("newton_hema", None),
    ("machine_cpu", None),
    ("triazines", 2),
]


# =========================
# ABLAÇÃO (gamma, t, contraste supervisionado)
# =========================
def _load_ablation_done_keys(path):
    key_cols = ["dataset", "seed", "sweep", "sweep_value"]
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, usecols=key_cols)
    except Exception:
        df = pd.read_csv(path)
    for c in key_cols:
        if c not in df.columns:
            return set()
    keys = set()
    for r in df[key_cols].itertuples(index=False, name=None):
        keys.add((str(r[0]), int(r[1]), str(r[2]), float(r[3])))
    return keys

def run_ablation():
    global DIFF_GAMMA  # sobrescrito temporariamente durante cada combinação

    os.makedirs(ABLATION_OUT_DIR, exist_ok=True)
    done_keys = _load_ablation_done_keys(ABLATION_RUNS_PATH)
    if done_keys:
        print(f"[ABLATION][RESUME] {len(done_keys)} combinações já feitas. Pulando o que já existe.")

    datasets = {}
    for name, did in ABLATION_DATASETS.items():
        X, y, meta = load_openml(did)
        datasets[name] = (X, y)
        print(f"[ABLATION][OK] {name}: X={X.shape}, classes={len(np.unique(y))}")

    combos = []
    for g in ABLATION_GAMMA_GRID:
        combos.append(("gamma", g, g, DIFF_STEPS, ETA_SAME, ETA_DIFF))
    for t in ABLATION_T_GRID:
        combos.append(("t_steps", t, DIFF_GAMMA, t, ETA_SAME, ETA_DIFF))
    for m in ABLATION_SUPERVISION_MULT_GRID:
        combos.append(("supervision_mult", m, DIFF_GAMMA, DIFF_STEPS, ETA_SAME * m, ETA_DIFF * m))

    for dataset_name, (X, y) in datasets.items():
        dataset_name_safe = safe_dirname(dataset_name)
        n = X.shape[0]
        n_classes = len(np.unique(y))
        k_graph = min(max(N_NEIGHBORS_BASE, K_PER_CLASS * n_classes), n - 1)
        k_cap = max(2, (n // 2) - 1)
        k_graph = min(k_graph, k_cap)
        k_trust = min(k_graph, n - 1)

        for sweep, sweep_value, gamma, t_steps, eta_same, eta_diff in combos:
            for seed in RANDOM_SEEDS:
                key = (dataset_name_safe, int(seed), sweep, float(sweep_value))
                if key in done_keys:
                    continue

                row = {
                    "dataset": dataset_name_safe, "seed": int(seed),
                    "sweep": sweep, "sweep_value": float(sweep_value),
                    "gamma": float(gamma), "t_steps": int(t_steps),
                    "eta_same": float(eta_same), "eta_diff": float(eta_diff),
                    "dim": int(ABLATION_DIM), "label_noise": float(ABLATION_NOISE),
                    "status": "OK", "error_msg": "",
                }
                try:
                    # gamma é usado dentro de build_knn_graph_and_scale via DIFF_GAMMA global;
                    # sobrescrevemos temporariamente para esta combinação e restauramos depois.
                    old_gamma = DIFF_GAMMA
                    DIFF_GAMMA = gamma
                    try:
                        A_sup, scale_s, alpha_same, alpha_diff = build_knn_graph_and_scale(
                            X, y, n_neighbors=k_graph, eta_same=eta_same, eta_diff=eta_diff
                        )
                    finally:
                        DIFF_GAMMA = old_gamma

                    Ddiff = diffusion_distance_from_affinity(A_sup, steps=t_steps)
                    emb = classical_mds_from_dist(Ddiff, n_components=ABLATION_DIM)

                    mets = compute_metrics(X, emb, y, seed=seed, n_neighbors_trust=k_trust)
                    cls = eval_classifiers(emb, y, seed=seed)
                    row.update(mets)
                    row.update(cls)
                    row["sei_scale_s"] = scale_s
                    row["alpha_same"] = alpha_same
                    row["alpha_diff"] = alpha_diff
                except Exception as e:
                    row["status"] = "ERROR"
                    row["error_msg"] = f"{type(e).__name__}: {e}"

                append_row_csv(row, ABLATION_RUNS_PATH)
                done_keys.add(key)
                print(f"[ABLATION][{dataset_name_safe}] sweep={sweep} value={sweep_value} seed={seed} -> {row['status']}")

    # Pós-processamento: médias + IC95%
    df = pd.read_csv(ABLATION_RUNS_PATH)
    df_ok = df[df["status"] == "OK"].copy()
    metrics_cols = ["trustworthiness", "distance_corr", "silhouette",
                     "KNN_acc", "SVM_acc", "RF_acc", "KNN_f1", "SVM_f1", "RF_f1"]

    summary_rows = []
    for (dataset, sweep, sweep_value), g in df_ok.groupby(["dataset", "sweep", "sweep_value"]):
        row = {"dataset": dataset, "sweep": sweep, "sweep_value": sweep_value}
        for mc in metrics_cols:
            m, s, lo, hi = mean_ci95(g[mc].values) if mc in g.columns else (np.nan,)*4
            row[f"{mc}_mean"], row[f"{mc}_std"] = m, s
            row[f"{mc}_ci95_lo"], row[f"{mc}_ci95_hi"] = lo, hi
        summary_rows.append(row)

    pd.DataFrame(summary_rows).sort_values(["dataset", "sweep", "sweep_value"]).to_csv(
        ABLATION_SUMMARY_PATH, index=False, encoding="utf-8"
    )
    print(f"\n✅ Ablação concluída.\n- Runs:    {ABLATION_RUNS_PATH}\n- Summary: {ABLATION_SUMMARY_PATH}")


# =========================
# COMPARAÇÃO: SUPERVISÃO LOCALMENTE ADAPTATIVA
# =========================
def _load_adaptive_done_keys(path):
    key_cols = ["dataset", "seed", "kappa"]
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    for c in key_cols:
        if c not in df.columns:
            return set()
    return {(str(r[0]), int(r[1]), float(r[2])) for r in df[key_cols].itertuples(index=False, name=None)}

def run_adaptive_comparison():
    os.makedirs(ABLATION_OUT_DIR, exist_ok=True)
    done_keys = _load_adaptive_done_keys(ADAPTIVE_RUNS_PATH)
    if done_keys:
        print(f"[ADAPTIVE][RESUME] {len(done_keys)} combinações já feitas. Pulando o que já existe.")

    datasets = {}
    for name, did in ABLATION_DATASETS.items():
        X, y, meta = load_openml(did)
        datasets[name] = (X, y)
        print(f"[ADAPTIVE][OK] {name}: X={X.shape}, classes={len(np.unique(y))}")

    # (kappa, adaptive): kappa=0.0 é o baseline com ADAPTIVE_SUPERVISION=False
    configs = [(0.0, False)] + [(float(kp), True) for kp in ADAPTIVE_KAPPA_GRID]

    for dataset_name, (X, y) in datasets.items():
        dataset_name_safe = safe_dirname(dataset_name)
        n = X.shape[0]
        n_classes = len(np.unique(y))
        k_graph = min(max(N_NEIGHBORS_BASE, K_PER_CLASS * n_classes), n - 1)
        k_cap = max(2, (n // 2) - 1)
        k_graph = min(k_graph, k_cap)
        k_trust = min(k_graph, n - 1)

        for kappa, adaptive in configs:
            for seed in RANDOM_SEEDS:
                key = (dataset_name_safe, int(seed), float(kappa))
                if key in done_keys:
                    continue

                row = {
                    "dataset": dataset_name_safe, "seed": int(seed),
                    "kappa": float(kappa), "adaptive": bool(adaptive),
                    "gamma": float(DIFF_GAMMA), "t_steps": int(DIFF_STEPS),
                    "eta_same": float(ETA_SAME), "eta_diff": float(ETA_DIFF),
                    "dim": int(ABLATION_DIM), "label_noise": float(ABLATION_NOISE),
                    "status": "OK", "error_msg": "",
                }
                try:
                    A_sup, scale_s, alpha_same, alpha_diff = build_knn_graph_and_scale(
                        X, y, n_neighbors=k_graph, eta_same=ETA_SAME, eta_diff=ETA_DIFF,
                        adaptive=adaptive, kappa=kappa
                    )
                    Ddiff = diffusion_distance_from_affinity(A_sup, steps=DIFF_STEPS)
                    emb = classical_mds_from_dist(Ddiff, n_components=ABLATION_DIM)

                    mets = compute_metrics(X, emb, y, seed=seed, n_neighbors_trust=k_trust)
                    cls = eval_classifiers(emb, y, seed=seed)
                    row.update(mets)
                    row.update(cls)
                    row["sei_scale_s"] = scale_s
                    row["alpha_same"] = alpha_same
                    row["alpha_diff"] = alpha_diff
                except Exception as e:
                    row["status"] = "ERROR"
                    row["error_msg"] = f"{type(e).__name__}: {e}"

                append_row_csv(row, ADAPTIVE_RUNS_PATH)
                done_keys.add(key)
                print(f"[ADAPTIVE][{dataset_name_safe}] kappa={kappa} adaptive={adaptive} seed={seed} -> {row['status']}")

    # Pós-processamento: médias + IC95%
    df = pd.read_csv(ADAPTIVE_RUNS_PATH)
    df_ok = df[df["status"] == "OK"].copy()
    metrics_cols = ["trustworthiness", "distance_corr", "silhouette",
                     "KNN_acc", "SVM_acc", "RF_acc", "KNN_f1", "SVM_f1", "RF_f1"]

    summary_rows = []
    for (dataset, kappa), g in df_ok.groupby(["dataset", "kappa"]):
        row = {"dataset": dataset, "kappa": kappa}
        for mc in metrics_cols:
            m, s, lo, hi = mean_ci95(g[mc].values) if mc in g.columns else (np.nan,)*4
            row[f"{mc}_mean"], row[f"{mc}_std"] = m, s
            row[f"{mc}_ci95_lo"], row[f"{mc}_ci95_hi"] = lo, hi
        summary_rows.append(row)

    pd.DataFrame(summary_rows).sort_values(["dataset", "kappa"]).to_csv(
        ADAPTIVE_SUMMARY_PATH, index=False, encoding="utf-8"
    )
    print(f"\n[OK] Comparação adaptativa concluída.\n- Runs:    {ADAPTIVE_RUNS_PATH}\n- Summary: {ADAPTIVE_SUMMARY_PATH}")


# =========================
# AVALIAÇÃO INDUTIVA (leakage-free out-of-sample)
# =========================
def _load_raw_for_inductive(data_id=None, name=None, version=None, subsample_max=MAX_SAMPLES_RUN):
    """Como load_openml/load_openml_by_name, mas SEM imputar/padronizar
    (isso é ajustado só no treino de cada split). As linhas subamostradas são
    as mesmas do pipeline principal (stratified_subsample só depende de y)."""
    if data_id is not None:
        ds = fetch_openml(data_id=data_id, as_frame=True)
    else:
        ds = fetch_openml(name=name, version=version, as_frame=True)
    Xo = pd.get_dummies(ds.data, dummy_na=True).to_numpy(dtype=np.float64)
    y = LabelEncoder().fit_transform(np.array(ds.target).ravel().astype(str))
    Xo, y = stratified_subsample(Xo, y, max_n=subsample_max, random_state=0)
    return Xo, y


def _inductive_datasets():
    specs = [(name, did, None, None) for name, did in DATASET_IDS.items()]
    specs += [("triazines_v2", None, "triazines", 2)]
    return [s for s in specs if s[0] not in INDUCTIVE_EXCLUDED]


def _fit_rbf_mapper(X_train, Y_train):
    """RBFInterpolator X->Y. Tenta thin_plate_spline exato; se falhar
    (pontos duplicados, ou n_train < p+1 exigido pelo polinômio de grau 1),
    cai para gaussian com smoothing=1e-6. Com pontos duplicados o TPS pode
    "resolver" um sistema quase singular sem erro e devolver lixo; por isso o
    MSE relativo no treino também é usado como critério de fallback."""
    try:
        mapper = RBFInterpolator(X_train, Y_train, kernel="thin_plate_spline")
        res = mapper(X_train) - Y_train
        rel = float(np.mean(res ** 2)) / (float(np.mean(np.var(Y_train, axis=0))) + 1e-12)
        if not np.isfinite(rel) or rel > 1e-6:
            raise np.linalg.LinAlgError(f"TPS mal condicionado (train_mse_rel={rel:.3g})")
        return mapper, "thin_plate_spline", ""
    except Exception as e:
        tps_err = f"{type(e).__name__}: {e}"
    D = pairwise_distances(X_train)
    med = float(np.median(D[np.triu_indices_from(D, k=1)])) + 1e-12
    eps = 1.0 / med
    mapper = RBFInterpolator(X_train, Y_train, kernel="gaussian", epsilon=eps, smoothing=1e-6)
    return mapper, "gaussian", tps_err


def _fit_mapper_krr(X_train, Y_train):
    """Kernel Ridge (kernel RBF, multi-saída) X->Y, agnóstico a rótulo.
    alpha/gamma por GridSearchCV 3-fold só no treino. KernelRidge não tem
    gamma="scale", então usa g0 = 1/(p * X_train.var()) e o grid [0.1, 1, 10]*g0.
    Diferente do RBFInterpolator, não interpola: train_mse > 0 é esperado."""
    g0 = 1.0 / (X_train.shape[1] * float(X_train.var()) + 1e-12)
    grid = {"alpha": [0.01, 0.1, 1.0, 10.0], "gamma": [0.1 * g0, g0, 10.0 * g0]}
    gs = GridSearchCV(KernelRidge(kernel="rbf"), grid, cv=3, n_jobs=1)
    gs.fit(X_train, Y_train)
    bp = gs.best_params_
    info = f"alpha={bp['alpha']:g}, gamma={bp['gamma']:.4g} (g0={g0:.4g}), cv_r2={gs.best_score_:.4f}"
    return gs.best_estimator_, "krr_rbf", info


def _mapper_type(method):
    return method.split("+", 1)[1].lower() if "+" in method else "none"


def _ensure_inductive_mapper_type_column(path):
    """Migração única: acrescenta a coluna mapper_type ao CSV de runs (backup antes)."""
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if "mapper_type" in df.columns:
        return
    bak = path + ".bak_before_mapper_type"
    if not os.path.exists(bak):
        df.to_csv(bak, index=False, encoding="utf-8")
    df["mapper_type"] = df["method"].map(_mapper_type)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[INDUCTIVE] coluna mapper_type adicionada ao CSV (backup: {bak})")


def _append_inductive_row(row, path):
    """Append alinhado ao header existente (colunas faltantes -> NaN)."""
    if os.path.exists(path):
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        extra = set(row) - set(cols)
        if extra:
            raise KeyError(f"colunas novas não previstas no CSV: {sorted(extra)}")
        pd.DataFrame([row]).reindex(columns=cols).to_csv(
            path, mode="a", index=False, header=False, encoding="utf-8")
    else:
        append_row_csv(row, path)


def _sei_nystrom_extension(A_sup, Ddiff, X_train, X_test, k_graph, n_components):
    """Extensão out-of-sample nativa do SEI-ISOMAP (Nyström/landmark MDS).

    Ponto de teste x: afinidade aos k vizinhos de treino com custo SEM termo
    supervisionado (y_test é desconhecido): a = exp(-DIFF_GAMMA * d). Linha de
    transição p(x) = a / sum(a); linha de difusão p_t(x) = p(x) P^(t-1).
    delta^2 = ||p_t(x) - P^t_j||^2 para todo j de treino, projetado no cMDS do
    treino pela fórmula de Gower: y = -0.5 Lambda^(-1/2) V^T (delta^2 - mu).
    Retorna Y_train (= classical_mds_from_dist), Z_test e o MSE de sanidade
    (projeção das próprias linhas P^t de treino, que deve reproduzir Y_train)."""
    N = X_train.shape[0]
    row_sum = np.maximum(np.array(A_sup.sum(axis=1)).ravel(), 1e-300)
    P = np.asarray(A_sup.multiply(1.0 / row_sum[:, None]).todense())
    Pt_1 = np.eye(N)                      # P^(t-1)
    for _ in range(DIFF_STEPS - 1):
        Pt_1 = Pt_1 @ P
    Pt = (P @ Pt_1).astype(np.float32)    # P^t, mesma precisão de diffusion_distance_from_affinity

    D2 = Ddiff.astype(np.float64) ** 2
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:n_components]
    lam_raw = np.clip(vals[idx], 0, None)
    # Componentes com autovalor ~0 (n_components > posto do cMDS, ex.: d=400 com N~480)
    # NÃO podem entrar na fórmula de Gower: dividir por sqrt(lam) pequeno amplifica ruído
    # numérico nos pontos de teste. Mantemos só autovalores >= 1% do maior (cauda = ruído
    # que o StandardScaler dos classificadores equalizaria); demais colunas ficam zeradas.
    keep = lam_raw > 1e-2 * max(lam_raw.max(), 1e-300)
    lam = np.where(keep, lam_raw, 1.0)
    V = vecs[:, idx] * keep[None, :]
    Y_train = V * np.sqrt(np.where(keep, lam_raw, 0.0))[None, :]
    mu = D2.mean(axis=0)

    def project(delta2):
        return -0.5 * (delta2 - mu[None, :]) @ V / np.sqrt(lam)[None, :]

    nbrs = NearestNeighbors(n_neighbors=k_graph).fit(X_train)
    dist, nn_idx = nbrs.kneighbors(X_test)
    A_te = np.zeros((X_test.shape[0], N))
    # log-sum-exp por linha: P_te é invariante ao deslocamento por linha; evita o colapso
    # (todos os pontos de teste no mesmo lugar) quando exp(-gamma*dist) underflowa.
    np.put_along_axis(A_te, nn_idx, np.exp(-_gamma_eff() * (dist - dist.min(axis=1, keepdims=True))), axis=1)
    P_te = A_te / np.maximum(A_te.sum(axis=1, keepdims=True), 1e-300)
    Pt_te = (P_te @ Pt_1).astype(np.float32)
    Z_test = project(pairwise_distances(Pt_te, Pt, metric="euclidean").astype(np.float64) ** 2)

    mse = float(np.mean((project(D2) - Y_train) ** 2))
    return Y_train, Z_test, mse


def _fit_eval_classifiers(F_train, y_train, F_test, y_test, seed):
    clfs = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=6, weights="distance")),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale")),
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }
    out = {}
    for nm, clf in clfs.items():
        pred = clf.fit(F_train, y_train).predict(F_test)
        out[f"{nm}_acc"] = float(accuracy_score(y_test, pred))
        out[f"{nm}_f1"] = float(f1_score(y_test, pred, average="macro"))
    return out


def run_inductive_eval():
    os.makedirs(ABLATION_OUT_DIR, exist_ok=True)
    done_keys = set()
    if os.path.exists(INDUCTIVE_RUNS_PATH):
        df_done = pd.read_csv(INDUCTIVE_RUNS_PATH)
        done_keys = {(str(r.dataset), int(r.seed), str(r.method)) for r in df_done.itertuples()}
        print(f"[INDUCTIVE][RESUME] {len(done_keys)} linhas já feitas. Pulando o que já existe.")

    methods = ["SEI-ISOMAP+RBF", "ISOMAP+RBF", "RAW",
               "SEI-ISOMAP+KNNREG", "ISOMAP+KNNREG",
               "SEI-ISOMAP+NYSTROM", "ISOMAP+NATIVE",
               "SEI-ISOMAP+KRR", "ISOMAP+KRR"]
    _ensure_inductive_mapper_type_column(INDUCTIVE_RUNS_PATH)

    for ds_name, did, oname, over in _inductive_datasets():
        ds_name = safe_dirname(ds_name)
        if all((ds_name, s, m) in done_keys for s in RANDOM_SEEDS for m in methods):
            continue
        try:
            Xo, y = _load_raw_for_inductive(data_id=did, name=oname, version=over)
        except Exception as e:
            print(f"[INDUCTIVE][PULOU] {ds_name} -> {type(e).__name__}: {e}")
            continue
        print(f"[INDUCTIVE][OK] {ds_name}: X={Xo.shape}, classes={len(np.unique(y))}")

        for seed in RANDOM_SEEDS:
            if all((ds_name, seed, m) in done_keys for m in methods):
                continue

            try:
                idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=INDUCTIVE_TEST_SIZE,
                                                  stratify=y, random_state=seed)
                split_mode = "stratified"
            except ValueError:
                # classe com 1 amostra: estratificação impossível
                idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=INDUCTIVE_TEST_SIZE,
                                                  random_state=seed)
                split_mode = "random"

            prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
            X_train = prep.fit_transform(Xo[idx_tr])
            X_test = prep.transform(Xo[idx_te])
            y_train, y_test = y[idx_tr], y[idx_te]

            n = X_train.shape[0]
            n_classes = len(np.unique(y_train))
            k_graph = min(max(N_NEIGHBORS_BASE, K_PER_CLASS * n_classes), n - 1)
            k_cap = max(2, (n // 2) - 1)
            k_graph = min(k_graph, k_cap)

            base = {
                "dataset": ds_name, "seed": int(seed), "dim": int(INDUCTIVE_DIM),
                "label_noise": float(INDUCTIVE_NOISE), "split_mode": split_mode,
                "n_train": int(n), "n_test": int(len(idx_te)), "p": int(X_train.shape[1]),
                "k_graph": int(k_graph), "gamma": float(DIFF_GAMMA), "t_steps": int(DIFF_STEPS),
                "eta_same": float(ETA_SAME), "eta_diff": float(ETA_DIFF),
            }

            emb_cache = {}

            def sei_train():
                # só (X_train, y_train); y_test nunca é usado aqui
                if "SEI" not in emb_cache:
                    A_sup, _, _, _ = build_knn_graph_and_scale(
                        X_train, y_train, n_neighbors=k_graph, eta_same=ETA_SAME, eta_diff=ETA_DIFF
                    )
                    emb_cache["SEI"] = (A_sup, diffusion_distance_from_affinity(A_sup, steps=DIFF_STEPS))
                return emb_cache["SEI"]

            def isomap_train():
                if "ISOMAP" not in emb_cache:
                    emb_cache["ISOMAP"] = safe_isomap(X_train, n_neighbors=k_graph, n_components=INDUCTIVE_DIM)
                return emb_cache["ISOMAP"]

            for method in methods:
                if (ds_name, seed, method) in done_keys:
                    continue
                # "rbf_kernel" guarda o nome do mapeador/kernel usado e
                # "rbf_fallback_reason" informação extra (ex.: hiperparâmetros do KRR).
                row = dict(base, method=method, status="OK", error_msg="", rbf_kernel="",
                           rbf_fallback_reason="", embed_mode="",
                           train_mse_sanity=np.nan, train_mse_rel=np.nan,
                           mapper_type=_mapper_type(method))
                try:
                    if method == "RAW":
                        F_train, F_test = X_train, X_test
                    else:
                        if method == "SEI-ISOMAP+NYSTROM":
                            A_sup, Ddiff = sei_train()
                            Y_train, F_test, mse = _sei_nystrom_extension(
                                A_sup, Ddiff, X_train, X_test, k_graph, INDUCTIVE_DIM
                            )
                            row["rbf_kernel"] = "nystrom_gower"
                        elif method == "ISOMAP+NATIVE":
                            iso = Isomap(n_neighbors=k_graph, n_components=INDUCTIVE_DIM, eigen_solver="dense")
                            Y_train = iso.fit_transform(X_train)
                            F_test = iso.transform(X_test)
                            mse = float(np.mean((iso.transform(X_train) - Y_train) ** 2))
                            row["rbf_kernel"] = "isomap_transform"
                        else:
                            if method.startswith("SEI-ISOMAP"):
                                Y_train = classical_mds_from_dist(sei_train()[1], n_components=INDUCTIVE_DIM)
                            else:
                                Y_train, row["embed_mode"] = isomap_train()
                            if method.endswith("+RBF"):
                                mapper, kernel, why = _fit_rbf_mapper(X_train, Y_train)
                                row["rbf_kernel"], row["rbf_fallback_reason"] = kernel, why[:300]
                                predict = mapper
                            elif method.endswith("+KRR"):
                                mapper, kernel, info = _fit_mapper_krr(X_train, Y_train)
                                row["rbf_kernel"], row["rbf_fallback_reason"] = kernel, info
                                predict = mapper.predict
                            else:  # +KNNREG
                                reg = KNeighborsRegressor(n_neighbors=min(N_NEIGHBORS_BASE, n - 1),
                                                          weights="distance").fit(X_train, Y_train)
                                row["rbf_kernel"] = "knn_regressor"
                                predict = reg.predict
                            mse = float(np.mean((predict(X_train) - Y_train) ** 2))
                            F_test = predict(X_test)
                        row["train_mse_sanity"] = mse
                        row["train_mse_rel"] = mse / (float(np.mean(np.var(Y_train, axis=0))) + 1e-12)
                        F_train = Y_train
                        if not np.isfinite(F_test).all():
                            raise FloatingPointError("Z_test não finito")
                    row.update(_fit_eval_classifiers(F_train, y_train, F_test, y_test, seed))
                except Exception as e:
                    row["status"] = "ERROR"
                    row["error_msg"] = f"{type(e).__name__}: {e}"[:500]

                _append_inductive_row(row, INDUCTIVE_RUNS_PATH)
                done_keys.add((ds_name, seed, method))
                extra = f" kernel={row['rbf_kernel']} mse={row['train_mse_sanity']:.2e}" if row["rbf_kernel"] else ""
                print(f"[INDUCTIVE][{ds_name}] seed={seed} {method} -> {row['status']}"
                      f"{' ' + row['error_msg'] if row['status'] != 'OK' else ''}{extra}")

    postprocess_inductive_eval()


def postprocess_inductive_eval():
    df = pd.read_csv(INDUCTIVE_RUNS_PATH)
    df_ok = df[df["status"] == "OK"].copy()
    metrics_cols = ["KNN_acc", "SVM_acc", "RF_acc", "KNN_f1", "SVM_f1", "RF_f1",
                    "train_mse_sanity", "train_mse_rel"]

    summary_rows = []
    for (dataset, method), g in df_ok.groupby(["dataset", "method"]):
        row = {"dataset": dataset, "method": method, "n_runs": int(len(g))}
        for mc in metrics_cols:
            m, s, lo, hi = mean_ci95(g[mc].values) if mc in g.columns else (np.nan,) * 4
            row[f"{mc}_mean"], row[f"{mc}_std"] = m, s
            row[f"{mc}_ci95_lo"], row[f"{mc}_ci95_hi"] = lo, hi
        summary_rows.append(row)
    df_sum = pd.DataFrame(summary_rows).sort_values(["dataset", "method"])
    df_sum.to_csv(INDUCTIVE_SUMMARY_PATH, index=False, encoding="utf-8")

    # Wilcoxon pareado por dataset (média das 10 seeds), SEI-ISOMAP+RBF vs ISOMAP+RBF
    df_ok["MEAN3_acc"] = df_ok[["KNN_acc", "SVM_acc", "RF_acc"]].mean(axis=1)
    per_ds = df_ok.groupby(["dataset", "method"])[["KNN_acc", "SVM_acc", "RF_acc", "MEAN3_acc"]].mean()

    lines = ["Avaliação indutiva (leakage-free): Wilcoxon pareado por dataset (two-sided),",
             "acurácia média das seeds por dataset. dim=2, label_noise=0.0, split 50/50.", ""]
    for a, b in [("SEI-ISOMAP+RBF", "ISOMAP+RBF"), ("SEI-ISOMAP+RBF", "RAW"),
                 ("SEI-ISOMAP+KRR", "ISOMAP+KRR"),
                 ("SEI-ISOMAP+KNNREG", "ISOMAP+KNNREG"), ("SEI-ISOMAP+NYSTROM", "ISOMAP+NATIVE"),
                 ("SEI-ISOMAP+NYSTROM", "SEI-ISOMAP+RBF"), ("SEI-ISOMAP+KNNREG", "SEI-ISOMAP+RBF")]:
        lines.append(f"=== {a} vs {b} ===")
        for col in ["KNN_acc", "SVM_acc", "RF_acc", "MEAN3_acc"]:
            try:
                sa = per_ds.xs(a, level="method")[col]
                sb = per_ds.xs(b, level="method")[col]
            except KeyError:
                lines.append(f"{col}: sem dados")
                continue
            common = sa.index.intersection(sb.index)
            diff = sa.loc[common] - sb.loc[common]
            wins, ties, losses = int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())
            try:
                p = float(wilcoxon(sa.loc[common], sb.loc[common]).pvalue)
            except ValueError:
                p = np.nan
            lines.append(f"{col}: n_datasets={len(common)} | wins/ties/losses={wins}/{ties}/{losses} "
                         f"| mean_diff={diff.mean():+.4f} | p={p:.4g}")
        lines.append("")
    with open(INDUCTIVE_WILCOXON_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[OK] Avaliação indutiva concluída.\n- Runs:     {INDUCTIVE_RUNS_PATH}"
          f"\n- Summary:  {INDUCTIVE_SUMMARY_PATH}\n- Wilcoxon: {INDUCTIVE_WILCOXON_PATH}")


# =========================
# MAIN
# =========================

# =========================================================
# PROTOCOLO PRINCIPAL LIVRE DE VAZAMENTO (--main-leakfree)
# =========================================================
# Para cada (dataset, seed, ruído, dim) e cada método, em validação cruzada
# estratificada de 5 folds:
#   1. o embedding (supervisionado ou não) é ajustado SOMENTE com o fold de treino
#      (X_treino e rótulos de treino, possivelmente ruidosos);
#   2. os pontos de teste são projetados por uma extensão out-of-sample
#      (nativa quando existe; Nyström para SEI-ISOMAP e SDM; regressor kNN para
#      Laplacian Eigenmaps, que não tem transform);
#   3. os classificadores são treinados com os rótulos (ruidosos) de treino e
#      avaliados nos rótulos LIMPOS do teste. Nenhum rótulo de teste é visto.
# Observação: a padronização (StandardScaler) é feita em todo o X antes da
# separação (estatísticas não supervisionadas, sem rótulos).
LEAKFREE_RUNS_PATH = os.path.join(OUT_DIR, "results_runs_leakfree.csv")
LEAKFREE_SUMMARY_PATH = os.path.join(OUT_DIR, "results_summary_leakfree.csv")
LEAKFREE_STATS_PATH = os.path.join(OUT_DIR, "stats_wilcoxon_leakfree.csv")
LEAKFREE_METHODS = ["SEI-ISOMAP", "SEI-NCA", "SDM", "NCA", "PCA", "ISOMAP", "LLE", "LAPLACIAN"]


def _sdm_nystrom(A, X_train, X_test, k_graph, n_components, t_steps):
    """SDM no treino + extensão de Nyström para o teste (afinidades sem termo supervisionado)."""
    N = X_train.shape[0]
    Ad = np.asarray(A.todense())
    dvec = np.maximum(Ad.sum(axis=1), 1e-300)
    S = Ad / np.sqrt(dvec)[:, None] / np.sqrt(dvec)[None, :]
    vals_all, vecs_all = np.linalg.eigh(S)
    order = np.argsort(vals_all)[::-1]
    k = min(n_components + 1, N - 1)
    vals = vals_all[order][1:k]
    vecs = vecs_all[:, order][:, 1:k]
    lam = np.clip(vals, 1e-12, None)
    Y_train = vecs * (lam ** t_steps)[None, :]

    nbrs = NearestNeighbors(n_neighbors=k_graph).fit(X_train)
    dist, nn_idx = nbrs.kneighbors(X_test)
    A_te = np.zeros((X_test.shape[0], N))
    # mesma escala do A de treino (que foi deslocado por _LAST_COST_SHIFT)
    np.put_along_axis(A_te, nn_idx, np.exp(-_gamma_eff() * (dist - _LAST_COST_SHIFT)), axis=1)
    dx = np.maximum(A_te.sum(axis=1, keepdims=True), 1e-300)
    S_te = A_te / np.sqrt(dx) / np.sqrt(dvec)[None, :]
    Z_test = (S_te @ vecs) * (lam ** (t_steps - 1))[None, :]
    return Y_train, Z_test


def _knn_extend(X_train, Y_train, X_test, k):
    reg = KNeighborsRegressor(n_neighbors=min(k, X_train.shape[0] - 1), weights="distance").fit(X_train, Y_train)
    return reg.predict(X_test)



# =========================
# SEI-NCA: difusão supervisionada (m dims) + projeção discriminante NCA (d dims)
# =========================
SEI_NCA_DIM = 20   # dimensão intermediária m das coordenadas de difusão (antes da projeção para d)


def sei_embed(X_tr, y_tr, X_te, n_components, k_graph, eta_mult=1.0):
    """SEI-ISOMAP puro (grafo supervisionado -> difusão -> cMDS) com extensão de Nyström-Gower.
    eta_mult escala (eta_same, eta_diff); eta_mult=0 remove o termo supervisionado."""
    A, _, _, _ = build_knn_graph_and_scale(X_tr, y_tr, n_neighbors=k_graph,
                                           eta_same=ETA_SAME * eta_mult, eta_diff=ETA_DIFF * eta_mult)
    Dd = diffusion_distance_from_affinity(A, steps=DIFF_STEPS)
    Ytr, Zte, _ = _sei_nystrom_extension(A, Dd, X_tr, X_te, k_graph, n_components)
    return Ytr, Zte


def _nca_reduce(F_tr, y_tr, F_te, d, seed, max_iter=100):
    """Projeção linear supervisionada (NCA) aprendida só no treino e aplicada ao teste (sem rótulos)."""
    try:
        nca = NeighborhoodComponentsAnalysis(n_components=d, random_state=seed, max_iter=max_iter).fit(F_tr, y_tr)
        return nca.transform(F_tr), nca.transform(F_te)
    except Exception:
        return F_tr[:, :d], F_te[:, :d]


def sei_nca_embed(X_tr, y_tr, X_te, d, k_graph, m=None, eta_mult=1.0, seed=0):
    """SEI-NCA: (1) SEI-ISOMAP em m dimensões; (2) NCA de m para d dimensões, ajustada no treino.
    No teste: coordenadas de Nyström (sem rótulos) -> mesma projeção NCA. Livre de vazamento."""
    m = min(SEI_NCA_DIM if m is None else m, X_tr.shape[0] - 2)
    m = max(m, d)
    Ytr, Zte = sei_embed(X_tr, y_tr, X_te, m, k_graph, eta_mult=eta_mult)
    return _nca_reduce(Ytr, y_tr, Zte, d, seed)


def _fold_embed(method, X_tr, y_tr, X_te, d, k_graph, seed):
    """Ajusta o método só com o treino; devolve (F_treino, F_teste, descrição da extensão)."""
    if method == "SEI-ISOMAP":
        A, _, _, _ = build_knn_graph_and_scale(X_tr, y_tr, n_neighbors=k_graph, eta_same=ETA_SAME, eta_diff=ETA_DIFF)
        Dd = diffusion_distance_from_affinity(A, steps=DIFF_STEPS)
        Ytr, Zte, _ = _sei_nystrom_extension(A, Dd, X_tr, X_te, k_graph, d)
        return Ytr, Zte, "nystrom_gower"
    if method == "SEI-NCA":
        Ytr, Zte = sei_nca_embed(X_tr, y_tr, X_te, d, k_graph, seed=seed)
        return Ytr, Zte, "nystrom_gower+nca"
    if method in ("SEI-NCA-S", "SEI-S"):   # versão com gamma = GAMMA_SCALE / s (invariante à escala)
        global GAMMA_MODE
        old_mode, GAMMA_MODE = GAMMA_MODE, "scale"
        try:
            if method == "SEI-NCA-S":
                Ytr, Zte = sei_nca_embed(X_tr, y_tr, X_te, d, k_graph, seed=seed)
            else:
                Ytr, Zte = sei_embed(X_tr, y_tr, X_te, d, k_graph)
        finally:
            GAMMA_MODE = old_mode
        return Ytr, Zte, "nystrom_gower" + ("+nca" if method == "SEI-NCA-S" else "")
    if method == "SEI0-NCA":      # ablação: sem termo supervisionado no grafo
        Ytr, Zte = sei_nca_embed(X_tr, y_tr, X_te, d, k_graph, eta_mult=0.0, seed=seed)
        return Ytr, Zte, "nystrom_gower+nca"
    if method == "ISOMAP-NCA":    # ablação: ISOMAP em m dims + NCA
        m = max(d, min(SEI_NCA_DIM, X_tr.shape[0] - 2))
        iso = Isomap(n_neighbors=k_graph, n_components=m, eigen_solver="dense").fit(X_tr)
        Ytr, Zte = _nca_reduce(iso.embedding_, y_tr, iso.transform(X_te), d, seed)
        return Ytr, Zte, "native+nca"
    if method == "PCA-NCA":       # ablação: PCA em m dims + NCA
        m = max(d, min(SEI_NCA_DIM, X_tr.shape[1], X_tr.shape[0] - 2))
        pca = PCA(n_components=m, random_state=seed).fit(X_tr)
        Ytr, Zte = _nca_reduce(pca.transform(X_tr), y_tr, pca.transform(X_te), d, seed)
        return Ytr, Zte, "native+nca"
    if method == "SDM":
        A, _, _, _ = build_knn_graph_and_scale(X_tr, y_tr, n_neighbors=k_graph, eta_same=ETA_SAME, eta_diff=ETA_DIFF)
        Ytr, Zte = _sdm_nystrom(A, X_tr, X_te, k_graph, d, DIFF_STEPS)
        return Ytr, Zte, "nystrom_sdm"
    if method == "NCA":
        nca = NeighborhoodComponentsAnalysis(n_components=d, random_state=seed).fit(X_tr, y_tr)
        return nca.transform(X_tr), nca.transform(X_te), "native"
    if method == "PCA":
        pca = PCA(n_components=d, random_state=seed).fit(X_tr)
        return pca.transform(X_tr), pca.transform(X_te), "native"
    if method == "ISOMAP":
        try:
            iso = Isomap(n_neighbors=k_graph, n_components=d, eigen_solver="dense").fit(X_tr)
            return iso.embedding_, iso.transform(X_te), "native"
        except ValueError:
            Ytr, _ = safe_isomap(X_tr, n_neighbors=k_graph, n_components=d)
            return Ytr, _knn_extend(X_tr, Ytr, X_te, N_NEIGHBORS_BASE), "knn_regressor(fallback)"
    if method == "LLE":
        k_lle = min(max(k_graph, d + 1), X_tr.shape[0] - 1)
        try:
            lle = LocallyLinearEmbedding(n_neighbors=k_lle, n_components=d, method="standard",
                                         eigen_solver="dense", random_state=seed).fit(X_tr)
            return lle.embedding_, lle.transform(X_te), "native"
        except Exception:
            Ytr, _ = safe_lle(X_tr, n_neighbors=k_lle, n_components=d, random_state=seed)
            return Ytr, _knn_extend(X_tr, Ytr, X_te, N_NEIGHBORS_BASE), "knn_regressor(fallback)"
    if method == "LAPLACIAN":
        try:
            Ytr = SpectralEmbedding(n_components=d, n_neighbors=k_graph, random_state=seed).fit_transform(X_tr)
        except Exception:
            Ytr = PCA(n_components=d, random_state=seed).fit_transform(X_tr)
        return Ytr, _knn_extend(X_tr, Ytr, X_te, N_NEIGHBORS_BASE), "knn_regressor"
    if method == "UMAP":
        um = umap.UMAP(n_components=d, n_neighbors=k_graph, min_dist=0.1, metric="euclidean",
                       random_state=seed).fit(X_tr)
        return um.embedding_, um.transform(X_te), "native"
    raise ValueError(f"método desconhecido: {method}")


def run_main_leakfree(quick=False, n_seeds=None):
    import warnings
    warnings.filterwarnings("ignore")
    seeds = RANDOM_SEEDS if n_seeds is None else RANDOM_SEEDS[:n_seeds]
    noises = [0.0, 0.2] if quick else LABEL_NOISE_RATES
    if quick and n_seeds is None:
        seeds = RANDOM_SEEDS[:3]
    methods = LEAKFREE_METHODS + (["UMAP"] if HAS_UMAP else [])
    done_keys = load_done_keys(LEAKFREE_RUNS_PATH)
    if done_keys:
        print(f"[LEAKFREE][RESUME] {len(done_keys)} linhas já feitas em {LEAKFREE_RUNS_PATH}")
    print(f"[LEAKFREE] seeds={seeds} ruídos={noises} métodos={methods}")

    datasets = {}
    for name, did in DATASET_IDS.items():
        try:
            datasets[name] = load_openml(did)
        except Exception as e:
            print(f"[LEAKFREE][PULOU-ID] {name} -> {type(e).__name__}: {e}")
    for name, ver in DATASET_SPECS:
        try:
            X_, y_, meta_ = load_openml_by_name(name=name, version=ver)
            tag = f"{meta_['openml_name']}_v{meta_['openml_version']}" if meta_["openml_version"] is not None else meta_["openml_name"]
            datasets[tag] = (X_, y_, meta_)
        except Exception as e:
            print(f"[LEAKFREE][PULOU-NOME] {name} -> {type(e).__name__}: {e}")

    for ds_name, (X, y, meta) in datasets.items():
        ds_name = safe_dirname(ds_name)
        n = X.shape[0]
        if n < 10:
            continue
        n_classes_all = len(np.unique(y))
        p = X.shape[1]
        max_d = min(p, int(0.8 * n) - 1)
        dims = {d for d in EMBED_DIMS if d <= max_d}
        if max_d >= 2:
            dims.add(max_d)
        dims = sorted(dims) or [2]
        print(f"\n[LEAKFREE] {ds_name} X={X.shape} classes={n_classes_all} dims={dims}")

        for seed in seeds:
            for noise in noises:
                nk = round(float(noise), 1)
                y_noisy = add_label_noise(y, nk, seed)
                try:
                    folds = list(StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=seed).split(X, y))
                except ValueError:
                    from sklearn.model_selection import KFold
                    folds = list(KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=seed).split(X))
                for d in dims:
                    for method in methods:
                        key = _coerce_key_tuple(ds_name, seed, nk, d, method)
                        if key in done_keys:
                            continue
                        row = {"status": "OK", "error_msg": "", "dataset": ds_name, "seed": int(seed),
                               "label_noise": float(nk), "dim": int(d), "method": method,
                               "protocol": "leakfree_5fold", "extension": "",
                               "openml_name": meta["openml_name"], "openml_id": meta["openml_id"],
                               "openml_version": meta["openml_version"], "subsample_max": meta["subsample_max"],
                               "n_original": meta["n_original"], "n_used": meta["n_used"],
                               "n_removed": meta["n_removed"], "reduction_pct": meta["reduction_pct"]}
                        accs = {k: [] for k in ["KNN_acc", "SVM_acc", "RF_acc", "KNN_f1", "SVM_f1", "RF_f1"]}
                        try:
                            for tr, te in folds:
                                X_tr, X_te = X[tr], X[te]
                                y_tr, y_te = y_noisy[tr], y[te]
                                n_tr = X_tr.shape[0]
                                kg = min(max(N_NEIGHBORS_BASE, K_PER_CLASS * len(np.unique(y_tr))), n_tr - 1)
                                kg = min(kg, max(2, (n_tr // 2) - 1))
                                F_tr, F_te, ext = _fold_embed(method, X_tr, y_tr, X_te, d, kg, seed)
                                if not (np.isfinite(F_tr).all() and np.isfinite(F_te).all()):
                                    raise FloatingPointError("embedding não finito")
                                res = _fit_eval_classifiers(F_tr, y_tr, F_te, y_te, seed)
                                for kk in accs:
                                    accs[kk].append(res[kk])
                                row["extension"] = ext
                            for kk in accs:
                                row[kk] = float(np.mean(accs[kk]))
                        except Exception as e:
                            row["status"] = "ERROR"
                            row["error_msg"] = f"{type(e).__name__}: {e}"[:400]
                            for kk in accs:
                                row[kk] = np.nan
                        append_row_csv(row, LEAKFREE_RUNS_PATH)
                        done_keys.add(key)
                        print(f"[LEAKFREE][{ds_name}] seed={seed} noise={nk} d={d} {method} -> {row['status']}"
                              f"{' ' + row['error_msg'] if row['status'] != 'OK' else ''}")

    postprocess_outputs(LEAKFREE_RUNS_PATH, LEAKFREE_SUMMARY_PATH, LEAKFREE_STATS_PATH, write_reduction=False)
    print(f"\n[LEAKFREE] ✅ Concluído. Runs: {LEAKFREE_RUNS_PATH}")


def main():
    # Carrega done_keys do CSV existente (resume)
    done_keys = load_done_keys(RUNS_PATH)
    if done_keys:
        print(f"[RESUME] Encontradas {len(done_keys)} linhas já feitas (OK/ERROR). Vou pular o que já existe no CSV.")
        print(f"[RESUME] CSV: {RUNS_PATH}")

    datasets = {}

    # IDs
    for name, did in DATASET_IDS.items():
        try:
            X, y, meta = load_openml(did)
            datasets[name] = (X, y, meta)
            print(f"[OK-ID] {name} -> {meta['openml_name']} (id={meta['openml_id']}) | N:{meta['n_original']}→{meta['n_used']} ({meta['reduction_pct']:.1f}%)")
        except Exception as e:
            print(f"[PULOU-ID] {name} (id={did}) -> {type(e).__name__}: {e}")

    # names
    for name, ver in DATASET_SPECS:
        try:
            X, y, meta = load_openml_by_name(name=name, version=ver)
            tag = f"{meta['openml_name']}_v{meta['openml_version']}" if meta["openml_version"] is not None else meta["openml_name"]
            datasets[tag] = (X, y, meta)
            print(f"[OK-NOME] {name} -> {tag} (id={meta['openml_id']}) | N:{meta['n_original']}→{meta['n_used']} ({meta['reduction_pct']:.1f}%)")
        except Exception as e:
            print(f"[PULOU-NOME] {name} (ver={ver}) -> {type(e).__name__}: {e}")

    # Salva reduction imediatamente (independente do experimento)
    reduction_rows = []
    for ds_name, (_, _, meta) in datasets.items():
        reduction_rows.append({
            "dataset": safe_dirname(ds_name),
            "openml_name": meta["openml_name"],
            "openml_id": meta["openml_id"],
            "openml_version": meta["openml_version"],
            "subsample_max": meta["subsample_max"],
            "n_original": meta["n_original"],
            "n_used": meta["n_used"],
            "n_removed": meta["n_removed"],
            "reduction_pct": meta["reduction_pct"],
        })
    df_red = pd.DataFrame(reduction_rows).sort_values("reduction_pct", ascending=False).reset_index(drop=True)
    df_red.to_csv(REDUCTION_PATH, index=False, encoding="utf-8")
    print(f"[INFO] dataset_reduction.csv salvo em: {REDUCTION_PATH}")

    # Executa (incremental)
    try:
        for ds_name, (X, y, meta) in datasets.items():
            run_dataset_incremental(ds_name, X, y, meta, done_keys)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Você interrompeu a execução. Tudo que já foi feito ficou salvo no CSV.")
        print("[INTERRUPT] Rode novamente para continuar de onde parou.")
        return

    # Pós-processa
    postprocess_outputs()

    print("\n✅ Concluído!")
    print(f"- Runs:       {RUNS_PATH}")
    print(f"- Summary:    {SUMMARY_PATH}")
    print(f"- Stats:      {STATS_PATH}")
    print(f"- Reduction:  {REDUCTION_PATH}")
    print(f"- Figuras:    {FIG_DIR}")

if __name__ == "__main__":
    if "--ablation" in sys.argv:
        run_ablation()
    elif "--adaptive-compare" in sys.argv:
        run_adaptive_comparison()
    elif "--inductive-eval" in sys.argv:
        run_inductive_eval()
    elif "--main-leakfree" in sys.argv:
        _ns = None
        if "--seeds" in sys.argv:
            _ns = int(sys.argv[sys.argv.index("--seeds") + 1])
        run_main_leakfree(quick=("--quick" in sys.argv), n_seeds=_ns)
    else:
        main()