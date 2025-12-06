
각 단어마다 hi를 하나씩 만듦.

hi마다 하나씩 qi , ki, vi를 Wq, Wk, Wv라는 (d_model x dk) 차원의 가중치 행렬로 projection하여 만듦. W는 학습가능한 파라미터 행렬.

q0 와 k0 ~ kN을 내적해서 query0 에 대한 softmax에 넣음.

그 softmax를 가중치로 써서 v0~vN을 가중합하여 Oo

