
# Memory cell

h만 있으면 지금 당장 무엇을 출력해야하는가? 를 정하는 연산 도중 원래 기억이 소실되어버리니까 맨 윗줄 memory cell을 추가해서 계속 가져감.

까먹어도 될 부분을 제거한 오래된 기억


# forget gate

h t-1과 xt를 보고 memory cell의 장기기억에서 지워도 될 내용을 지움.

# input gate

1. i필터: h t-1과 xt를 보고 이번 새로운 내용을 추가할지 결정.
2. C필터: tanh (-1 ~ 1)로 이 정보가 어떤 내용인지 결정.

# output gate

