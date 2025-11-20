

### ADAM (Mini batch + RMS prop + momentum + bias correction)


# 학습 단계

multiclass classification 이라고 하면, x1...xn이 입력으로 주어짐.

one-hot vector가 정답 y로 주어진게 샘플 1개. 

<국룰 조합을 가정>
Loss function : Cross entropy (여기 한정 KL과 같음) 선택하고, 
hidden layer activation function : ReLU
output layer activation function: softmax 선택

현재가 K번째 학습이라고 가정하면,
미니배치로 샘플 32개 or 64개 정도를 동시에 넣고, 

첫번째 레이어에서 z = wx + b 계산하고, (물론 여기서는 BN 과정이 있어서 b는 의미를 잃으므로 아예 정의하지 않는 것 좋다.)
Batch Normalization 시행: 이번 배치의 32개 샘플의 평균과 분산(분산 + eps)으로 평균 0 표준편차 1로 정규화 한 다음에 k-1번째 스텝에서 정해진 scale parameter (gamma)랑 shift parameter (beta)를 활용해 맛있는 위치로 보내고, ReLU에다가 넣기.
이 과정 중 뒤에서 몰래 나중에 추론 단계에서 사용할 'Running Mean / Running Variance'를 저장함.

두번째 레이어에서도 똑같이 z = wx + b계산하고, (물론 여기서는 BN 과정이 있어서 b는 의미를 잃으므로 아예 정의하지 않는 것이 좋다.)
Batch Normalization 시행: 이번 배치의 32개 샘플의 평균과 분산(분산 + eps)으로 평균 0 표준편차 1로 정규화 한 다음에 k-1번째 스텝에서 정해진 scale parameter (gamma)랑 shift parameter (beta)를 활용해 맛있는 위치로 보내고, ReLU에다가 넣기.
이 과정 중 뒤에서 몰래 나중에 추론 단계에서 사용할 'Running Mean / Running Variance'를 저장함.

이를 마지막 레이어까지 반복한 후, 

마지막 출력층 layer에서 softmax로 꺼내고, 정답으로 주어진 y와 출력층에서 꺼낸 softmax값을 Cross Entropy Loss function에 넣고,

back propagation 실행: gradient를 오직 '계산'만 해놓고, 

Momentum method 실행: k - 1 번째 1st moment(m_k-1) 와 back propagation으로 계산한 이번 gradient를 학습전에 미리 정해놓은 비율 (Beta 1, 약 0.9)로 가중합하여 이번 스텝의 1st moment(m_k)를 구하고, 이번 1st moment를 저장. 

RMS Prop 실행: k - 1번째 2nd moment(v_k-1)와 back propagation으로 계산한 이번 gradient의 제곱을 학습 전에 미리 정해놓은 비율 (Beta 2, 약 0.999)로 가중합하여 이번 스텝의 2nd moment(v_k)를 구하고, 이번 2nd moment를 저장.

bias correction 1: 이번 1st moment를 1 - (Beta1)^k 으로 나눠 스케일링. m_k(hat) 얻음.
bias correction 2: 이번 2nd moment를 1 - (Beta2)^k 으로 나눠 스케일링. v_k(hat) 얻음.

마지막으로 gradient 스텝 밟기: w_{k+1} = w_k - eta * ( m_k(hat) / sqrt(v_k(hat) + eps) )
여기서 w는 모든 weight M개와  
모든 BN layer의 γ, β 파라미터(총합 2·N개)를 일렬로 나열한 column vector이다.  
(N은 BN이 적용되는 모든 채널/뉴런의 총합)

이 전체 과정을 여러 epoch 동안 반복하면서  
training loss / validation loss / metric이  
수렴하거나 더 이상 개선되지 않을 때까지 학습을 진행.





