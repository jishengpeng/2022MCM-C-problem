import pandas as pd
import csv
import numpy as np

df=pd.read_csv("C题处理后的中间文件2.csv")
height,weight=df.shape

# height=10

m=df.values

huangjin=[]

for i in range(0,height):
    if(m[i][2]==0):
        huangjin.append(m[i][3])



k=1 #k代表现在持有的是黄金还是比特币
n=0 #黄金统计

rateG=0.01
rateB=0.02

oldprice_G=0
oldprice_B=0
newprice_G=0
newprice_B=0
target=1000

res=[]
#变成现金0
#变成比特币1
#变成黄金2

for i in range(1,height-1):
    res1=[]
    print("第%d天的资产为：%f"%(i,target))
    # res1.append(m[i][0])
    # res1.append(target)
    # res.append(res1)
    #如果黄金不能交易
    if(m[i][2]==1):
        #持现
        if(k==0):
            oldprice_B=m[i][1]
            newprice_B=m[i+1][1]
            tmp=target /(1+rateB) * newprice_B / oldprice_B
            tmp=round(tmp,2)
            target1=max(target,tmp)
            if(target1==target):
                k1=0
            else:
                k1=1


        #持比特币
        if(k==1):
            oldprice_B = m[i][1]
            newprice_B = m[i + 1][1]
            tmp1=target*(newprice_B/oldprice_B)
            tmp1=round(tmp1,2)
            tmp2=target/(1+rateB)
            tmp2=round(tmp2,2)
            target1=max(tmp1,tmp2)
            if(target1==tmp2):
                k1=0
            else:
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
        oldprice_B = m[i][1]
        newprice_B = m[i + 1][1]
        oldprice_G=huangjin[n]
        newprice_G=huangjin[n+1]

        if(k==0):
            tmp1=target/(1+rateG)*newprice_G/oldprice_G
            tmp1=round(tmp1,2)
            tmp2=target/(1+rateB)*newprice_B/oldprice_B
            tmp2=round(tmp2,2)
            target1=max(tmp1,tmp2,target)
            if(target1==target):
                k1=0
            if(target1==tmp1):
                k1=2
            if(target1==tmp2):
                k1=1


        if(k==1):
            tmp1=target*newprice_B/oldprice_B
            # print(tmp1)
            tmp1=round(tmp1,2)
            tmp2=target/(1+rateB)
            tmp2=round(tmp2,2)
            tmp3=target/(1+rateB+rateG)*newprice_G/oldprice_G
            tmp3=round(tmp3,3)
            target1=max(tmp1,tmp2,tmp3)
            if(target1==tmp1):
                k1=1
            if(target1==tmp2):
                k1=0
            if(target1==tmp3):
                k1=2


        if(k==2):
            tmp1 = target * newprice_G / oldprice_G
            tmp1=round(tmp1, 2)
            tmp2 = target /(1+rateG)
            tmp2 = round(tmp2, 2)
            tmp3 = target /(1+rateB+rateG) * newprice_B / oldprice_B
            tmp3 = round(tmp3, 3)
            target1 = max(tmp1, tmp2, tmp3)
            if (target1 == tmp1):
                k1 = 2
            if (target1 == tmp2):
                k1 = 0
            if (target1 == tmp3):
                k1 = 1

        n=n+1
        res1.append(m[i][0])
        res1.append(k1)
        res.append(res1)
        k=k1
        target=target1

res=np.array(res)
print(res.shape[0])
# res1=[]
# res1.append(m[height-1][0])
# res1.append(target)
# res.append(res1)
print("第%d天的资产为：%f"%(height-1,target))

# print(res)
f=open('每天的实际操作.csv','w',encoding='utf-8',newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["日期(月/日/年)","操作"])
csv_writer.writerow(["9/11/16",0])
for i in range(0,1824):
    csv_writer.writerow(res[i])
f.close()





