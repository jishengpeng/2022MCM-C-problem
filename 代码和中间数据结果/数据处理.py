import pandas
import pandas as pd
import csv

df1=pd.read_csv("BCHAIN-MKPRU.csv")
df2=pd.read_csv("LBMA-GOLD.csv")
height,weight=df1.shape
# height=10
x1=df1.values
x2=df2.values
res=[]
j=0#用来标记新的文件
for i in range(0,height):
    tmp=[]
    if(x1[i][0]==x2[j][0] ):
        if(pandas.isnull(x2[j][1])):
            tmp.append(x1[i][0])
            tmp.append(x1[i][1])
            tmp.append(1)
            res.append(tmp)
            j=j+1
        else:
            tmp.append(x1[i][0])
            tmp.append(x1[i][1])
            tmp.append(0)       #0代表可以售卖
            tmp.append(x2[j][1])
            j=j+1
            res.append(tmp)
    else:
        tmp.append(x1[i][0])
        tmp.append(x1[i][1])
        tmp.append(1)
        res.append(tmp)

#将这些数据保存在CSV文件中
f=open('C题处理后的中间文件2.csv','w',encoding='utf-8',newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["日期(月/日/年)","比特币价值","是否可以买卖黄金","黄金价值"])
for i in range(0,height):
    csv_writer.writerow(res[i])
f.close()


