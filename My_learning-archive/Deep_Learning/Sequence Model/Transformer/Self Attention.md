
# self attention

각 단어마다 hi를 하나씩 만듦.

hi마다 하나씩 qi , ki, vi를 Wq, Wk, Wv라는 (d_model x dk) 차원의 가중치 행렬로 projection하여 만듦. W는 학습가능한 파라미터 행렬.

q0 와 k0 ~ kN을 내적한 후 root(d_k)로 나누어 분산을 1로 맞추고 
query0 에 대한 softmax에 넣음.
그 softmax를 가중치로 써서 v0~vN을 가중합하여 O0을 얻음.

q1과 k0 ~ kN을 내적해서 query 1에 대한...

이하 동문. O1을 얻음. 계속 반복.
(물론 벡터로 설명했지만 실제로는 행렬로 한 번에 처리됨.)



# multi - head attention

d_model = d_k x H (head개수) 임. (보통 이렇게 설계함)

위에서 얻은 O1, O2... OH들을 열 방향으로 concatenate 해서

위로 단어길이 L 만큼 쌓고, dk씩 H개가 붙은 L x (d_k x H) 차원의 O_concat 행렬 생성.

- 마지막 FC (output projection)
이제 한 번 더 **선형변환**으로 head들을 섞어준다: W_O 행렬 (d_model x d_model)로
O_concat x W_O (L x d_model)