import pandas as pd
import csv
import numpy as np

df=pd.read_csv("C题处理后的中间文件2.csv")
height,weight=df.shape

# height=10

df1=pd.read_excel("比特币的预测价格.xls")
df1=np.array(df1.values)

# print(df1)

df4=[]
for i in range(1,df1.shape[0]):
    tmp=[]
    tmp.append(df1[i][0])
    df4.append(tmp)

df4=np.array(df4)
df1=df4
# print(df1)

df2=pd.read_excel("黄金的预测价格.xls")
df2=np.array(df2.values)

df3=[]
df3.append([1324.6])
for i in range(0,df2.shape[0]):
    df3.append(df2[i])

df3=np.array(df3)
# print(df3)

m=df.values

huangjin=[]

for i in range(0,height):
    if(m[i][2]==0):
        huangjin.append(m[i][3])
res=[]


k=1 #k代表现在持有的是黄金还是比特币
n=0 #黄金统计

rateG=0.01
rateB=0.02

oldprice_G=0
oldprice_B=0
newprice_G=0
newprice_B=0
target=1000
#
# # res=[]

height=height-1
#
for i in range(1,height-1):
    res1=[]
    print("第%d天的资产为：%f"%(i,target))

    #如果黄金不能交易
    if(m[i][2]==1):
        #持现
        if(k==0):
            oldprice_B=df1[i-1][0]
            newprice_B=df1[i][0]
            tmp=target /(1+rateB) * newprice_B / oldprice_B
            tmp=round(tmp,2)
            target1=max(target,tmp)
            if(target1==target):
                target1=target
                k1=0
            else:
                target1=round(target /(1+rateB) * m[i+1][1]/ m[i][1],2)
                k1=1


        #持比特币
        if(k==1):
            oldprice_B = df1[i-1][0]
            newprice_B = df1[i][0]
            tmp1=target*(newprice_B/oldprice_B)
            tmp1=round(tmp1,2)
            tmp2=target/(1+rateB)
            tmp2=round(tmp2,2)
            target1=max(tmp1,tmp2)
            if(target1==tmp2):
                k1=0
                target1 = round(target / (1 + rateB), 2)
            else:
                target1=round(target*(m[i+1][1]/m[i][1]),2)
                k1=1


        #持黄金
        if(k==2):
            target1=target
            k1=2

        res1.append(m[i][0])
        res1.append(k1)
        res.append(res1)
        k=k1
        target=target1

    if(m[i][2]==0):
        oldprice_B1 = m[i][1]
        newprice_B1 = m[i + 1][1]
        oldprice_G1=huangjin[n]
        newprice_G1=huangjin[n+1]

        oldprice_B = df1[i-1][0]
        newprice_B = df1[i][0]
        oldprice_G=df3[n][0]
        newprice_G=df3[n+1][0]

        if(k==0):
            tmp1=target/(1+rateG)*newprice_G/oldprice_G
            tmp1=round(tmp1,2)
            tmp2=target/(1+rateB)*newprice_B/oldprice_B
            tmp2=round(tmp2,2)
            target1=max(tmp1,tmp2,target)
            if(target1==target):
                target1=target
                k1=0
            else:
                if(target1==tmp1):
                    target1=round(target/(1+rateG)*newprice_G1/oldprice_G1,2)
                    k1=2
                else:
                    target1=round(target/(1+rateB)*newprice_B1/oldprice_B1,2)
                    k1=1


        if(k==1):
            tmp1=target*newprice_B/oldprice_B
            # print(tmp1)
            tmp1=round(tmp1,2)
            tmp2=target/(1+rateB)
            tmp2=round(tmp2,2)
            tmp3=target/(1+rateB+rateG)*newprice_G/oldprice_G
            # print(tmp3)
            tmp3=round(tmp3,2)
            target1=max(tmp1,tmp2,tmp3)
            if(target1==tmp1):
                target1=target*newprice_B1/oldprice_B1
                k1=1
            else:
                if(target1==tmp2):
                    target1=round(target / (1 + rateB),2)
                    k1=0
                else:
                    k1=2
                    target1=round(target/(1+rateB+rateG)*newprice_G1/oldprice_G1,2)


        if(k==2):
            tmp1 = target * newprice_G / oldprice_G
            tmp1=round(tmp1, 2)
            tmp2 = target /(1+rateG)
            tmp2 = round(tmp2, 2)
            tmp3 = target /(1+rateB+rateG) * newprice_B / oldprice_B
            tmp3 = round(tmp3, 3)
            target1 = max(tmp1, tmp2, tmp3)
            if (target1 == tmp1):
                target1=round(target * newprice_G1 / oldprice_G1,2)
                k1 = 2
            else:
                if (target1 == tmp2):
                    k1 = 0
                    target1=round(target /(1+rateG),2)
                else:
                    k1 = 1
                    target1=round(target /(1+rateB+rateG) * newprice_B1 / oldprice_B1,2)
        res1.append(m[i][0])
        res1.append(k1)
        res.append(res1)
        n=n+1
        k=k1
        target=target1

print("第%d天的资产为：%f"%(height-1,target))

res=np.array(res)
print(res.shape[0])

f=open('最终版每天的实际操作.csv','w',encoding='utf-8',newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["日期(月/日/年)","操作"])
csv_writer.writerow(["9/11/16",0])
for i in range(0,1823):
    csv_writer.writerow(res[i])
f.close()





