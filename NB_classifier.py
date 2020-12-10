"""
File name:     NB_classifier.py
Authors:       Maximilien Fathi
Date:          December 13, 2020
Description:   
"""

import pandas as pd
from collections import Counter

df = pd.read_csv("covid_training.tsv", sep='\t', header=None)

# for i in df.columns:
#     df[i] = df[i].str.lower().str.split()
# # print(df[1].values[0]) # counts column names

# NB_BOW_OV = Counter()
# df[1].str.lower().str.split().apply(NB_BOW_OV .update)
# print(NB_BOW_OV )

temp  = Counter()
df[1].str.lower().str.split().apply(temp .update)
NB_BOW_FV = {k:v for k,v in temp.items() if v >= 2}
print(type(NB_BOW_FV))

# print(df[1].values[7])


