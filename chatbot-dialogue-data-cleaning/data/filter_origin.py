import pandas as pd

origin = pd.read_csv("origin_prob.csv", encoding="utf-8")
labels = origin["Label"]
indexs = list()
indexs_surplus = list()
for index, value in enumerate(labels):
    if value > 0.5:
        indexs.append(index)
    else:
        indexs_surplus.append(index)

origin_f = origin.iloc[indexs]
origin_ff = origin_f[["Context", "Utterance"]]
origin_ff.to_csv("origin1.csv", index=False, encoding="utf-8")
origin_f.to_csv("upper.csv", index=False, encoding="utf-8")

origin_s = origin.iloc[indexs_surplus]
origin_s.to_csv("surplus.csv", index=False, encoding="utf-8")

