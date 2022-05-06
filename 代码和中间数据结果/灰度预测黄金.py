import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

def GM11(x,n):
    '''
    灰色预测
    x：序列，numpy对象
    n:需要往后预测的个数
    '''
    x1 = x.cumsum()#一次累加
    z1 = (x1[:len(x1) - 1] + x1[1:])/2.0#紧邻均值
    z1 = z1.reshape((len(z1),1))
    B = np.append(-z1,np.ones_like(z1),axis=1)
    Y = x[1:].reshape((len(x) - 1,1))
    #a为发展系数 b为灰色作用量
    [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)#计算参数
    result = (x[0]-b/a)*np.exp(-a*(n-1))-(x[0]-b/a)*np.exp(-a*(n-2))
    S1_2 = x.var()#原序列方差
    e = list()#残差序列
    for index in range(1,x.shape[0]+1):
        predict = (x[0]-b/a)*np.exp(-a*(index-1))-(x[0]-b/a)*np.exp(-a*(index-2))
        e.append(x[index-1]-predict)
    S2_2 = np.array(e).var()#残差方差
    C = S2_2/S1_2#后验差比
    if C<=0.35:
        assess = '后验差比<=0.35，模型精度等级为好'
    elif C<=0.5:
        assess = '后验差比<=0.5，模型精度等级为合格'
    elif C<=0.65:
        assess = '后验差比<=0.65，模型精度等级为勉强'
    else:
        assess = '后验差比>0.65，模型精度等级为不合格'
    #预测数据
    predict = list()
    for index in range(x.shape[0]+1,x.shape[0]+n+1):
        predict.append((x[0]-b/a)*np.exp(-a*(index-1))-(x[0]-b/a)*np.exp(-a*(index-2)))
    predict = np.array(predict)
    return {
            'a':{'value':a,'desc':'发展系数'},
            'b':{'value':b,'desc':'灰色作用量'},
            'predict':{'value':result,'desc':'第%d个预测值'%n},
            'C':{'value':C,'desc':assess},
            'predict':{'value':predict,'desc':'往后预测%d个的序列'%(n)},
            }

if __name__ == "__main__":



    # #首先进行比特币的灰度预测
    df=pd.read_csv("C题处理后的中间文件2.csv")
    train=[]
    height,weight=df.shape
    # height=10
    sum=0
    for i in range(0,height):
        if(df.values[i][2]==0):
            train.append(df.values[i][3])
            sum=sum+1
    train=np.array(train)
    print(train)
    res=[]
    tmp=[]
    tmp.append(train[0])
    tmp.append(0)
    res.append(tmp)
    tmp=[]
    tmp.append(train[1])
    tmp.append(0)
    res.append(tmp)
    tmp=[]
    tmp.append(train[2])
    tmp.append(0)
    res.append(tmp)
    # print(res)
    for i in range(2,sum-1):
        x=train[0:i+1]
        y=train[i+1:i+2]
        result = GM11(x,len(y))
        predict = result['predict']['value']
        predict = np.round(predict,1)
        tmp=[]
        tmp.append(predict[0])
        tmp.append(round(abs(predict[0]-train[i+1]),2))
        # print('真实值:',y)
        # print('预测值:',predict)
        # print(result)
        res.append(tmp)

    # #计算误差
    # for i in range(0,height):
    #     wucha1=abs(res[i]-train[i])
    #     wucha.append(wucha1)

    # print(1)
    #将其保存在csv文件中终

    #生成最终的一列一列
    ans=[]
    j=0
    for i in range(0,height):
        tmp=[]
        if(df.values[i][2]==1):
            tmp.append(df.values[i][0])
            tmp.append(df.values[i][2])
            ans.append(tmp)
        else:
            tmp.append(df.values[i][0])
            tmp.append(df.values[i][2])
            tmp.append(df.values[i][3])
            tmp.append(res[j][0])
            tmp.append(res[j][1])
            j = j + 1
            ans.append(tmp)

    # print(res)
    f = open('灰度预测黄金.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["日期(月/日/年)","是否可以买卖黄金","黄金价值","黄金预测价值","误差"])
    for i in range(0,height):
        csv_writer.writerow(ans[i])
    f.close()

