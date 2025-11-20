## 학습

multiclass 라고 하면, x1...xn이 입력으로 주어지고, one-hot vector가 y 정답으로 주어진게 샘플 1개이고, 
Loss function으로 KL을 선택하고, 
input layer activation function으로 ReLU선택하고,
output layer activation function으로 softmax를 선택하는 
국룰 조합으로 가서, 

### SGD

이 샘플 한개를 먹여서 
가중치 업데이트하고 gradient 한 스텝 밟고, 
다른 거 한개맥여서 가중치 조정하고 한스텝밟고 
반복


### mini-batch


1. **Batch (32개) 투입**
    
2. **Linear ($Wx+b$)** $\rightarrow$ 값들이 엉망진창
    
3. **BN Layer** $\rightarrow$ **강제로 줄 세우고($\mu, \sigma$), 적절히 재배치($\gamma, \beta$) + 족보 작성**
    
4. **Activation (ReLU/Softmax)** $\rightarrow$ 깔끔하게 비선형성 추가
    
5. **Loss 계산 (Cross Entropy)**
    
6. **Backpropagation** $\rightarrow$ $W, b$ 뿐만 아니라 **$\gamma, \beta$도 같이 업데이트**
    
7. **(반복)**