"""Combine the original four-configuration search with the eight further configurations of
tune_ext.py and select, per architecture, the configuration with the highest mean
internal-validation correlation over the two tuning traits -> best_hp_rev.json, tuning_all.csv"""
import json
import pandas as pd
import models_dl as D
from tune_ext import EXT
import config as C
o = pd.read_csv(C.ROOT / "results" / "tuning.csv")
e = pd.concat([pd.read_csv(C.ROOT / "results" / f"tuning_ext_{a}.csv") for a in ("MLP", "CNN", "TRF")])
allt = pd.concat([o, e], ignore_index=True)
allt.to_csv(C.ROOT / "results" / "tuning_all.csv", index=False)
best = {}
for a in ("MLP", "CNN", "TRF"):
    m = allt[allt.arch == a].groupby("grid").val_r.mean()
    gi = int(m.idxmax())
    best[a] = (D.GRID[a] + EXT[a])[gi]
    print(a, "best grid", gi, round(m.max(), 3), best[a], "| original best", round(m[m.index < 4].max(), 3))
json.dump(best, open(C.ROOT / "best_hp_rev.json", "w"), indent=2)
