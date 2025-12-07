
# Memory cell

h만 있으면 지금 당장 무엇을 출력해야하는가? 를 정하는 연산 도중 원래 기억이 소실되어버리니까 맨 윗줄 memory cell을 추가해서 계속 가져감.

1. 까먹어도 될 부분을 제거한 오래된 기억
2. 새로운, 쓸만한 내용
을 단순히 더함.

# forget gate

h t-1과 xt를 보고 memory cell의 장기기억에서 지워도 될 내용을 지움.

# input gate

1. i필터: h t-1과 xt를 보고 이번 새로운 내용을 추가할지 결정.
2. C필터: tanh (-1 ~ 1)로 이 정보가 어떤 내용인지 결정.

# output gate

지금 당장 고려해야할 부분을 찝어내는 역할.

과거 의견 w1h(t-1) + 현재 의견w2x(t) + b0로 '투표' 하듯
더함.
이후 시그모이드로 통과시켜서 비선형성 + 둘 사이의 관계성 확보 

Ct를 tanh로 -1 ~ 1에 꾹 눌러담아서
저 시그모이드와 원소별 곱.


# peephole connection

원래 h(t-1), x(t)를 concatenate 했었는데,
여기서 원본 C(t-1)도 맨 왼쪽에 같이함.

raw memory도 같이 보는게 좋대


# bidirectional LSTM

LSTM은 순방향으로만 학습이 가능함.

나는 좋아한다. 콜라를,

같은 도치 구문들을 학습하기 쉽지않음.
그래서 양방향을 씀.

## bidirectional LSTM의 문제

- 실시간성 부족: 역방향으로도 학습해야하기 때문에, 문장 전체를 입력 받고 나서야 역방향 학습이 가능함.
- 계산 비용: $t-1$의 계산이 끝나야 $t$의 계산을 시작할 수 있는 순환(Recurrent) 구조


# ✔ Batch Normalization은 안씀

- LayerNorm (가장 많이 씀)
    
- WeightNorm
    
- 혹은 normalization 없이 기본 LSTM