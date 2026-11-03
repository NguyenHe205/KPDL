import pandas as pd
df = pd.read_csv("data/rh5k.csv", usecols=["idx","status","seconds","kappa","theta_bar","nu","rho","V0","H"])
df["fail"] = df.status != "ok"
print(df.status.value_counts())
for c in ["H", "V0", "theta_bar", "nu", "kappa", "rho"]:
    print(c); print(df.groupby(pd.cut(df[c], 5), observed=True).fail.mean().round(2))
print(df[df.status == "ok"].seconds.describe())