import pandas as pd, shutil
# remove do bench36_runs.csv as linhas geradas com o código ANTIGO (sem correção numérica),
# para que sejam refeitas com o código atual. Guarda backup.
APAGAR = ["SEI-NCA", "SEI0-NCA", "SEI-ISOMAP", "SEI-ISOMAP+RBF"]
shutil.copy("bench36_runs.csv", "bench36_runs_antes_limpar_sei.csv")
d = pd.read_csv("bench36_runs.csv")
n0 = len(d)
d = d[~d.method.isin(APAGAR)]
d.to_csv("bench36_runs.csv", index=False)
print(f"linhas: {n0} -> {len(d)} (removidas {n0-len(d)})")
print(d.groupby("method").size())