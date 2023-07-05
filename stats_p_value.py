import pandas as pd
from scipy.stats import mannwhitneyu

df1 = pd.read_csv('path')
df2 = pd.read_csv('path2')

for col in df1.colunms:
    _, p = mannwhitneyu(df1[col], df2[col])
    print(f"{f'{col}' : <30} {p}")
