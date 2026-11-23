# -*- coding: utf-8 -*-
"""Diagnóstico do SEI-ISOMAP em datasets onde ele colapsa (mfeat-pixel, vowel).
Uso:  python diag_sei.py mfeat-pixel      (ou: vowel, optdigits ...)"""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
import benchmark36 as B, sei_isomap as S
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

name = sys.argv[1] if len(sys.argv) > 1 else "mfeat-pixel"
spec = [t for t in B.BENCH36 if t[0] == name][0]
X, y, meta = B.load_bench_dataset(*spec, max_n=int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
print(name, X.shape, "classes", len(np.unique(y)), "n_train", len(ytr))
for k in (10, int(round(np.sqrt(len(ytr)))), 30):
    for t in (1, 3, 10):
        S.DIFF_STEPS = t
        A, s, _, _ = S.build_knn_graph_and_scale(Xtr, ytr, n_neighbors=k, eta_same=S.ETA_SAME, eta_diff=S.ETA_DIFF)
        nc, _ = connected_components(A, directed=False)
        Dd = S.diffusion_distance_from_affinity(A, steps=t)
        Ytr, Zte, _ = S._sei_nystrom_extension(A, Dd, Xtr, Xte, k, 2)
        acc_tr = KNeighborsClassifier(5).fit(Ytr, ytr).score(Ytr, ytr)
        acc_te = KNeighborsClassifier(5).fit(Ytr, ytr).score(Zte, yte)
        print(f"k={k:3d} t={t:2d} componentes={nc:3d} std_tr={Ytr.std(0).round(3)} std_te={Zte.std(0).round(3)} "
              f"kNN(treino,resub)={acc_tr:.3f} kNN(teste)={acc_te:.3f}")
from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=int(round(np.sqrt(len(ytr)))), n_components=2).fit(Xtr)
print("ISOMAP kNN(teste) =", round(KNeighborsClassifier(5).fit(iso.embedding_, ytr).score(iso.transform(Xte), yte), 3))
