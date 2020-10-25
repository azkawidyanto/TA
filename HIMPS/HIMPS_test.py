import math
import cProfile
import pstats
import io
import pandas as pd
import numpy as np
from pstats import SortKey

file = input('Masukkan Nama File: ')
k = input("Masukkan K yang diinginkan:")
data = pd.read_csv(file)
df = pd.DataFrame(
    data, columns=["ncall", "Function", "Support_plus", "Support_neg", "IG", "DS"])
pd.set_option('mode.chained_assignment', None)


def support(df):
    for i in range(0, len(df.ncall)):
        if df.ncall[i] == 1:
            df["Support_neg"][i] = 1
            df["Support_plus"][i] = 0
        else:
            df["Support_neg", i] = - \
                (math.ceil(df.ncall[i] / len(df.ncall)))
            df["Support_plus", i] = abs(
                len(df.Function) - abs(df.Support_neg[i]))


def total_transac(df):
    return sum(df.ncall)


def neg_transac(df):
    return (len(df.Function))


def pos_transac(df):
    return(total_transac(df) - neg_transac(df))


def H(a, b):
    if (a == 0) or (b == 0):
        return 0
    else:
        x = a / (a + b)
        y = b / (a + b)
        return (-(x * math.log(x, 2)) - (y * math.log(y, 2)))


def information_gain(df):
    a = total_transac(df)
    for i in range(0, len(df.ncall)):
        b = df.Support_plus[i] + df.Support_neg[i]
        c = (a-b) / a
        x = H(abs(pos_transac(df)), abs(neg_transac(df)))
        y = b * H(df.Support_plus[i], abs(df.Support_neg[i])) / a
        z = c * H(abs(pos_transac(df)) -
                  df.Support_plus[i], abs(neg_transac(df)) - df.Support_neg[i])
    df["IG", i] = abs(x-y-z)


def disc_Significance(df):
    for i in range(0, len(df["ncall"])):
        df.DS[i] = df.IG[i]


def topK_signature(df, k):
    df.sort_values(by=["DS"], inplace=True, ascending=False)


# result = pd.DataFrame(df, columns=["Function", "DS"])
# def disc_Significance(df):
#     for i in range(0, len(df.ncall)):
#         a = df.Support_neg[i]/abs(neg_transac(df))
#         b = df.Support_plus[i] / abs(pos_transac(df))
#         if (df.Support_neg[i] == 0) or (df.Support_plus[i] == 0):
#             df.DS[i] = 0
#         else:
#             if (a > b):
#                 df.DS[i] = information_gain(df)
#             else:
#                 df.DS[i] = 0
support(df)
information_gain(df)
disc_Significance(df)
topK_signature(df, k)

# df = pd.DataFrame(df, columns=["Function", "DS"])
print(df)
