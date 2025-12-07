
# self attention

각 단어마다 hi를 하나씩 만듦.

hi마다 하나씩 qi , ki, vi를 Wq, Wk, Wv라는 (d_model x dk) 차원의 가중치 행렬로 projection하여 만듦. W는 학습가능한 파라미터 행렬이며, 같은 self - attention layer에 있는 W는 모든 단어에 대해 같은 W를 씀. 

q0 와 k0 ~ kN을 내적한 후 root(d_k)로 나누어 분산을 1로 맞추고,
Causal masking 을 적용함.
query0에 대한 softmax에 넣음. 
그 softmax를 가중치로 써서 v0~vN을 가중합하여 O0을 얻음.

- X_att (Q, K, V) = softmax(QK^T / root(d_k) + M) * V

q1과 k0 ~ kN을 내적해서 query 1에 대한...
이하 동문. O1을 얻음. 계속 반복.
(물론 벡터로 설명했지만 실제로는 행렬로 한 번에 처리됨.)



# multi - head attention 

d_model = d_k x H (head개수) 임. (보통 이렇게 설계함)

각 self - attention에서 얻은 O1, O2... OH들을 열 방향으로 concatenate 해서

위로 단어길이 L 만큼 쌓고, dk씩 H개가 붙은 L x (d_k x H) 차원의 O_concat 행렬 생성.

- 마지막 FC (output projection)
이제 한 번 더 **선형변환**으로 head들을 섞어준다: W_O 행렬 (d_model x d_model)로
X_att = O_concat x W_O 
최종 출력 결과 : X_att: L x d_model

이게 곧장 ADD & NORM으로 들어감.


# 직관적인 이해

CNN이 커널 하나당 특징 하나 담당해서 conv layer 여러 겹 쌓으면 특징들끼리 결합되는거처럼, Attention블록도 여러번쌓으면 집중해야하는 포인트들 끼리 결합되어서, 
" I는 주어인데, 뒤에 and you가 나오면 뒤에 is가 아니라 are을 쓴다" 
같은 복합적인 특징이 추출되는 것.