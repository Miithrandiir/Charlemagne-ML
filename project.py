import pandas
import pandas as pd
import numpy as np

dataframe = pd.read_csv("./data/result_new.csv", sep=";")

print(dataframe.head())

res = pandas.DataFrame(columns=["cp_white", "cp_black", "player", "diff_level"])

for index, row in dataframe.iterrows():
    colorPLay = 0
    cps_white = row["cp_white"].split(' ')
    cps_black = row["cp_black"].split(' ')

    for i in range(min(len(cps_white), len(cps_black))):
        if cps_white[i] == '' or cps_black[i] == '':
            continue
        # np.concatenate(res, [cps_white[1], cps_black[2], "white" if colorPLay == 0 else "black", row["diff_level"]])
        dic= {"cp_white": cps_white[i], "cp_black": cps_black[i], "player": colorPLay,
             "diff_level": row["diff_level"]}
        res = pd.concat([res,  pandas.DataFrame.from_records([dic])])
        # res = res.append(
        #     {"cp_white": cps_white[1], "cp_black": cps_black[2], "player": ("white" if colorPLay == 0 else "black"),
        #      "diff_level": row["diff_level"]}, ignore_index=True)
        colorPLay = 0 if colorPLay == 1 else 1

print(res.to_csv("export_new.csv",index=False))