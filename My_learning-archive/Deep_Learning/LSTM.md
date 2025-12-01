
# Memory cell

h만 있으면 지금 당장 무엇을 출력해야하는가? 를 정하는 연산 도중 원래 기억이 소실되어버리니까 맨 윗줄 memory cell을 추가해서 계속 가져가는 거야.


# forget gate

h t-1과 xt를 보고 memory cell의 장기기억에서 지워도 될 내용을 지움.

# input gate

1. i필터: h t-1과 xt를 보고 새로운 정보를 추가할지 정함.
2. C필터: tanh (-1 ~ 1)로 이 정보가 어떤 내용을 결정

# output gate