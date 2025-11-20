## 학습

multiclass 라고 하면, x1...xn이 입력으로 주어지고, one-hot vector가 y 정답으로 주어진게 샘플 1개이고, 
Loss function으로 KL을 선택하고, 
input layer activation function으로 ReLU선택하고,
output layer activation function으로 softmax를 선택하는 국룰 조합으로 가서, 


### ADAM (Mini batch, RMS)


#### 학습 준비 단계

multiclass 라고 하면, x1...xn이 입력으로 주어짐.

one-hot vector가 정답 y로 주어진게 샘플 1개. 

Loss function : Cross entropy (여기 한정 KL과 같음) 선택하고, 
input layer activation function : ReLU
output layer activation function: softmax 선택
-----> 국룰 조합임.

#### 학습 단계

샘플 32, 64개 정도 한 번에 넣음.

forward 계산 z = wx + b (아직 뒤죽박죽)

Batch Normalization 실행 --> 뒤죽박죽인 z 정규화 (0,1)


그들의 평균 LOSS로 gradient descent 딱 Step 실행.


## !!디테일!!

Batch Normalization (BN)에서 업데이트하는 베타와 감마도 LOSS 함수의 변수이다.

따라서 LOSS 함수의 총 차원 수 = M (가중치 개수 = 간선 수) + 2 x N ( 뉴런개수 )