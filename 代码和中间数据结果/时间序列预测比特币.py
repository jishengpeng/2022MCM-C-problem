import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from d2l import torch as d2l
import torch
import torch.nn as nn

ji1=np.array(b_pred_linear).reshape(-1,1)
ji1=np.array(ji1)
ji2=Data.iloc[2:days_fit+1,1]
ji2=np.array(ji2)
book = xlwt.Workbook(encoding="utf-8", style_compression=0)
sheet = book.add_sheet("回归预测比特币", cell_overwrite_ok=True)
col = ("真实值","预测值")  # 元组,如果需要院校简介另加
for i in range(0, 2):
    sheet.write(0, i, col[i])
sheet.write(1,0,621.25)
sheet.write(1,1,621.25)
sheet.write(2,0,609.67)
sheet.write(2,1,609.67)

ji1=np.array(b_pred_linear).reshape(-1,1)
ji1=np.array(ji1)
ji2=Data.iloc[2:days_fit+1,1]
ji2=np.array(ji2)
book = xlwt.Workbook(encoding="utf-8", style_compression=0)
sheet = book.add_sheet("回归预测比特币", cell_overwrite_ok=True)
col = ("真实值","预测值")  # 元组,如果需要院校简介另加
for i in range(0, 2):
    sheet.write(0, i, col[i])
sheet.write(1,0,621.25)
sheet.write(1,1,621.25)
sheet.write(2,0,609.67)
sheet.write(2,1,609.67)
for i in range(0, 1824):
    print("第%d条" % (i + 1))
    sheet.write(i+3,0,ji1[i+2][0][0])
    sheet.write(i+3,1,ji2[i])
book.save("/home/jishengpeng/美赛的模拟练习/回归预测比特币.xls")



f=open(path+'/lstm比特币预测.csv','w',encoding='utf-8',newline="")
csv_writer=csv.writer(f)
csv_writer.writerow(["实际价格","预测价格"])
for i in range(0,input_size-100+1):
    tmp=[]
    tmp.append(actuals[i])
    tmp.append(round(predictions[i],2))
    csv_writer.writerow(tmp)
f.close()


path="/home/jishengpeng/美赛的模拟练习"
BCHAIN_MKPRU=pd.read_csv(path+"/BCHAIN-MKPRU.csv",dtype={"Date":np.str,"Value":np.float64})
LBMA_GOLD=pd.read_csv(path+"/LBMA-GOLD.csv",dtype={"Date":np.str,"Value":np.float64})
Data=pd.read_csv(path+"/C题处理后的中间文件2.csv")