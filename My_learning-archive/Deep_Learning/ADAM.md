## 학습

multiclass 라고 하면, x1...xn이 입력으로 주어지고, one-hot vector가 y 정답으로 주어진게 샘플 1개이고, 
Loss function으로 KL을 선택하고, 
input layer activation function으로 ReLU선택하고,
output layer activation function으로 softmax를 선택하는 국룰 조합으로 가서, 


### ADAM (Mini batch + RMS prop + momentum + bias correction)


#### 학습 준비 단계

multiclass 라고 하면, x1...xn이 입력으로 주어짐.

one-hot vector가 정답 y로 주어진게 샘플 1개. 

Loss function : Cross entropy (여기 한정 KL과 같음) 선택하고, 
input layer activation function : ReLU
output layer activation function: softmax 선택
-----> 국룰 조합임.

#### 학습 단계

multiclass 라고 하면, x1...xn이 입력으로 주어짐.

one-hot vector가 정답 y로 주어진게 샘플 1개. 

Loss function : Cross entropy (여기 한정 KL과 같음) 선택하고, 
hidden layer activation function : ReLU
output layer activation function: softmax 선택

현재가 K번째 학습이라고 가정하면,
미니배치로 샘플 32개 or 64개 정도를 동시에 넣고, 

첫번째 레이어에서 z = wx + b 계산하고, 
Batch Normalization 시행: 평균 0 표준편차 1인 분포로 정규화 한 다음에 k-1번째 스텝에서 정해진 베타랑 감마로 맛있는 위치로 보내고, ReLU에다가 넣기.

두번째 레이어에서도 똑같이 z = wx + b계산하고, 
Batch Normalization 시행: 평균 0 표준편차 1인 분포로 정규화 한 다음에 k-1번째 스텝에서 정해진 베타랑 감마로 맛있는 위치로 보내고, ReLU에 넣기... 

를 마지막 레이어까지 반복한 후, 

마지막 출력층 layer에서 softmax로 꺼내고, 정답으로 주어진 y와 출력층에서 꺼낸 softmax값을 Cross Entropy Loss function에 넣고,

back propagation 실행: gradient를 오직 '계산'만 해놓고, 

Momentum method로 k - 1 번째 1st moment 와 back propagation으로 계산한 이번 gradient를 학습전에 미리 정해놓은 비율 (Beta 1, 약 0.9) 로 가중합하여 이번 스텝의 1st moment를 구하고, 이번 1st moment를 저장한 후,
bias correction in Momentum method  : 이번 1st moment를 1 - (Beta1)^k 으로 나눠 스케일링.

RMS Prop으로 : k - 1번째 2nd moment와 back propagation으로 계산한 이번 gradient의 제곱을 학습 전에 미리 정해놓은 비율 (Beta 2, 약 0.999) 로 가중합하여 이번 스텝의 2nd moment를 구하고, 이번 2nd moment를 저장
bias correction in RMS Prop : 이번 2nd moment를 1 - (Beta2)^k 으로 나눠 스케일링.

마지막으로 gradient 스텝 밟기: w_{k+1} = w_k - eta * ( m_k / sqrt(v_k + eps) )


이 전체 과정을 gradient가 적정수준이상으로 작아질 때까지 반복.





## !!디테일!!

Batch Normalization (BN)에서 업데이트하는 베타와 감마도 LOSS 함수의 변수이다.

따라서 LOSS 함수의 총 차원 수 = M (가중치 개수 = 간선 수) + 2 x N ( 뉴런개수 )