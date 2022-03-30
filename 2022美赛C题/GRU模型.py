import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from d2l import torch as d2l
import torch
import torch.nn as nn
import csv
path="/home/jishengpeng/美赛的模拟练习"
BCHAIN_MKPRU=pd.read_csv(path+"/BCHAIN-MKPRU.csv",dtype={"Date":np.str,"Value":np.float64})
LBMA_GOLD=pd.read_csv(path+"/LBMA-GOLD.csv",dtype={"Date":np.str,"Value":np.float64})
Data=pd.read_csv(path+"/C题处理后的中间文件2.csv")

def to_timestamp(date):
    return int(time.mktime(time.strptime(date,"%m/%d/%y")))

#将日期变为自然数
start_timestamp=to_timestamp(Data.iloc[0,0])
for i in range(Data.shape[0]):
    Data.iloc[i,0]=(to_timestamp(Data.iloc[i,0])-start_timestamp)/86400
print(Data)

batch_size=1 # 应该只能为1
start_input=30
input_size=Data.shape[0]#训练：通过前input_size天预测input_size+1天，预测：通过2到input_size+1天预测第input_size+2天
hidden_size=20
# input_size=200
output_size=1
layers_size=3
lr=10
num_epochs=1000


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers_size):
        super().__init__()
        self.GRU_layer = nn.GRU(input_size, hidden_size, layers_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.GRU_layer(x)
        x = self.linear(x)
        return x

device=torch.device("cuda")

gru=GRUModel(30, hidden_size, output_size, layers_size).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(gru.parameters(), lr)

ji=np.array(Data.iloc[0:input_size,3].dropna())
input_size=ji.shape[0]-2

trainB_x=torch.from_numpy(ji[input_size-30:input_size].reshape(-1,batch_size,30)).to(torch.float32).to(device)
trainB_y=torch.from_numpy(ji[input_size].reshape(-1,batch_size,output_size)).to(torch.float32).to(device)

losses = []

for epoch in range(num_epochs):
    output = gru(trainB_x).to(device)
    loss = criterion(output, trainB_y)
    losses.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss" + str(epoch) + ":", loss.item())

# 预测，以比特币为例
# pred_x_train=torch.from_numpy(np.array(Data.iloc[1:input_size+1,1]).reshape(-1,1,input_size)).to(torch.float32).to(device)
pred_x_train=torch.from_numpy(ji[input_size-29:input_size+1]).reshape(-1,1,30).to(torch.float32).to(device)
pred_y_train=gru(pred_x_train).to(device)
print("prediction:",pred_y_train.item())
print("actual:",ji[input_size+1])

# 预测代码
losses = []
predictions = []
actuals = []
for i in range(start_input, input_size + 1):
    print("进行到input_size=", i)
    # gru=GRUModel(i, hidden_size, output_size, layers_size).to(device)
    gru = GRUModel(30, hidden_size, output_size, layers_size).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(gru.parameters(), lr)

    # 数据，以比特币为例
    trainB_x = torch.from_numpy(ji[i - 30:i].reshape(-1, batch_size, 30)).to(torch.float32).to(device)
    trainB_y = torch.from_numpy(ji[i].reshape(-1, batch_size, output_size)).to(torch.float32).to(device)

    loss = None

    for epoch in range(num_epochs):
        output = gru(trainB_x).to(device)
        loss = criterion(output, trainB_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("loss"+str(epoch)+":", loss.item())
    losses.append(loss.item())

    # 预测，以比特币为例
    pred_x_train = torch.from_numpy(ji[i - 29:i + 1].reshape(-1, 1, 30)).to(torch.float32).to(device)
    pred_y_train = gru(pred_x_train).to(device)
    # print("prediction:",pred_y_train.item())
    # print("actual:",Data.iloc[i+1,1])
    predictions.append(pred_y_train.item())
    actuals.append(ji[i + 1])
plt.plot(losses)

plt.plot(predictions)
plt.plot(actuals)

print(np.array(predictions).shape[0])
print(np.array(actuals).shape[0])
print(input_size-29)

f=open(path+'/周期lstm黄金预测1000版本.csv','w',encoding='utf-8',newline="")
csv_writer=csv.writer(f)
csv_writer.writerow(["实际价格","预测价格"])
for i in range(0,input_size-29):
    tmp=[]
    tmp.append(actuals[i])
    tmp.append(round(predictions[i],2))
    csv_writer.writerow(tmp)
f.close()