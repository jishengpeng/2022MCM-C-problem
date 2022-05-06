import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
# from d2l import torch as d2l
import torch
import torch.nn as nn
import xlwt

BCHAIN_MKPRU=pd.read_csv("BCHAIN-MKPRU.csv",dtype={"Date":np.str,"Value":np.float64})
LBMA_GOLD=pd.read_csv("LBMA-GOLD.csv",dtype={"Date":np.str,"Value":np.float64})
Data=pd.read_csv("C题处理后的中间文件2.csv")
df=pd.read_csv("C题处理后的中间文件2.csv")

def to_timestamp(date):
    return int(time.mktime(time.strptime(date,"%m/%d/%y")))

#将日期变为自然数
start_timestamp=to_timestamp(Data.iloc[0,0])
for i in range(Data.shape[0]):
    Data.iloc[i,0]=(to_timestamp(Data.iloc[i,0])-start_timestamp)/86400

days_fit=Data.shape[0] # 最小为2

bFit=Data.iloc[0:days_fit,0:2]
gFit=Data.iloc[0:days_fit,0::3].dropna() # 需要考虑NaN的问题


bitcoin_reg=linear_model.LinearRegression()
gold_reg=linear_model.LinearRegression()

bitcoin_reg.fit(np.array(bFit.iloc[:,0]).reshape(-1,1),np.array(bFit.iloc[:,1]).reshape(-1,1))
gold_reg.fit(np.array(gFit.iloc[:,0]).reshape(-1,1),np.array(gFit.iloc[:,1]).reshape(-1,1))

# print("bitcoin:",bitcoin_reg.predict(np.array([days_fit]).reshape(-1,1)))
# print("gold:",gold_reg.predict(np.array([days_fit]).reshape(-1,1)))


# 预测代码：
b_pred_linear = [None, None]
g_pred_linear = [None, None]
for day_fit in range(2, days_fit + 1):
    # print("进行到day_fit=",day_fit)
    bFit = Data.iloc[0:day_fit, 0:2]
    gFit = Data.iloc[0:day_fit, 0::3].dropna()

    bitcoin_reg = linear_model.LinearRegression()
    gold_reg = linear_model.LinearRegression()

    bitcoin_reg.fit(np.array(bFit.iloc[:, 0]).reshape(-1, 1), np.array(bFit.iloc[:, 1]).reshape(-1, 1))
    gold_reg.fit(np.array(gFit.iloc[:, 0]).reshape(-1, 1), np.array(gFit.iloc[:, 1]).reshape(-1, 1))

    b_pred_linear.append(bitcoin_reg.predict(np.array([day_fit]).reshape(-1, 1)))
    g_pred_linear.append(gold_reg.predict(np.array([day_fit]).reshape(-1, 1)))

ji1=np.array(b_pred_linear).reshape(-1,1)
ji1=np.array(ji1)
ji2=Data.iloc[2:days_fit+1,1]
ji2=np.array(ji2)

ji3=[]

for i in range(2,1826):
    ji3.append(round(ji1[i][0][0][0],2))

ji3=np.array(ji3)


# print(ji1[3][0][0][0])
book = xlwt.Workbook(encoding="utf-8", style_compression=0)
sheet = book.add_sheet("回归预测比特币", cell_overwrite_ok=True)
col = ("日期","预测值","真实值","误差")  # 元组,如果需要院校简介另加
for i in range(0, 4):
    sheet.write(0, i, col[i])
sheet.write(1,0,"9/11/16")
sheet.write(1,1,621.25)
sheet.write(1,2,621.25)
sheet.write(1,3,0)

sheet.write(2,0,"9/12/16")
sheet.write(2,1,609.67)
sheet.write(2,2,609.67)
sheet.write(2,3,0)
for i in range(0, 1824):
    print("第%d条" % (i + 1))
    sheet.write(i+3,0,df.values[i+2][0])
    sheet.write(i+3,1,ji3[i])
    sheet.write(i+3,2,ji2[i])
    sheet.write(i+3,3,abs(ji3[i]-ji2[i]))
book.save("回归预测比特币.xls")


