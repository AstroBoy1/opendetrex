ids = [];probs = [];xmin = [];ymin = [];xmax = [];ymax = []

for p in lines:
    ps = p.split(" ")
    ids.append(ps[0])
    probs.append(ps[1])
    xmin.append(ps[2])
    ymin.append(ps[3])
    xmax.append(ps[4])
    ymax.append(ps[5])

import pandas as pd

df = pd.DataFrame();df["ids"] = ids;df["probs"] = probs;df["xmin"] = xmin;df["ymin"] = ymin;df["xmax"] = xmax; df["ymax"] = ymax
df["ids"] = df["ids"].astype('str')

df.to_csv("agnostic_edges_test_predictions.csv")
