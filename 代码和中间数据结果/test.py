import pandas as pd
import numpy as np

df=pd.read_csv("C题处理后的中间文件2.csv")
height=df.shape[0]
sum=0
for i in range(0,height):
    if(df.values[i][2]==0):
        sum=sum+1
print(sum)