multiclass 라고 하면, x1...xn이 입력으로 주어지고, one-hot vector가 y 정답으로 주어진게 샘플 1개이고, 
SGD는 Loss function으로 KL을 선택하고, 
activation function으로 softmax를 선택하는 국룰조합으로 가서, 이 샘플 한개를 먹여서 가중치 업데이트하고 gradient 한 스텝 밟고, 다른거한개맥여서 가중치 조정하고 한스텝밟고 반복하는게 학습단계인가