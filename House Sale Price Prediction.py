#colab連結雲端硬碟
from google.colab import drive
drive.mount('/content/drive')

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


# 讀檔案
#train訓練用
data=pd.read_csv('/content/drive/MyDrive/ntut-ml-regression-2021/train-v3.csv')
X_train_raw = data.iloc[:,2:].values #第2列後為特徵值
y_train = data.iloc[:,1].values #第1列為要預測的房價

#test用來對答案
data2=pd.read_csv('/content/drive/MyDrive/ntut-ml-regression-2021/valid-v3.csv')
X_test_raw = data2.iloc[:,2:].values
y_test = data2.iloc[:,1].values

# 特徵縮放，讓x裡面每一行的值都變得差不多
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# 將資料轉為 tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# 模型建立
#上一層輸出和下一層輸入大小需一致
model = nn.Sequential(
    nn.Linear(21, 40),
    nn.ReLU(),
    nn.Linear(40, 80),
    nn.ReLU(),
    nn.Linear(80, 160),
    nn.ReLU(),
    nn.Linear(160, 150),
    nn.ReLU(),
    nn.Linear(150, 70),
    nn.ReLU(),
    nn.Linear(70, 35),
    nn.ReLU(),
    nn.Linear(35, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# loss function 對資料做優化
loss_fn = nn.MSELoss()  # 均方誤差(Mean square error，MSE)
#optimizer 優化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 300 #訓練類神經網路次數
batch_size = 10 #每10個資料分一堆訓練
batch_start = torch.arange(0, len(X_train), batch_size)


best_mse = np.inf
best_weights = None

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # 取分堆的資料
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            
            # forward pass
            y_pred = model(X_batch)
            
            #loss
            loss = loss_fn(y_pred, y_batch)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # update weights
            optimizer.step()
            
            # print progress
            bar.set_postfix(mse=float(loss))
            
    # 評估訓練的模型
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    #更新best_mse、best_weights
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# 儲存的 state_dict 參數代替原本初始的參數
model.load_state_dict(best_weights)

model.eval()

#完成模型訓練，預測房價
with torch.no_grad():
  d_test = pd.read_csv('/content/drive/MyDrive/ntut-ml-regression-2021/test-v3.csv') 
  data_test = d_test.iloc[:,1:].values
  X_sample = scaler.transform(data_test)
  X_sample = torch.tensor(X_sample, dtype=torch.float32)
  y_pred = model(X_sample)

df_sampleSubmission = pd.read_csv('/content/drive/MyDrive/ntut-ml-regression-2021/sampleSubmission.csv')
df_sampleSubmission['price'] = y_pred
df_sampleSubmission.to_csv('/content/drive/MyDrive/ntut-ml-regression-2021/sampleSubmission.csv', index=False)