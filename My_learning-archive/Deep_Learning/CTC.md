
# Connectionist Temporal Classification


CAT을 맞춘다고하자.

일단 빈칸을 사이사이에 끼워넣어 expansion함
()C()A()T()  --> 2*L + 1개로 늘어남

나중에 연속 중복 문자는 1개로 축소시킴


매 시간 T마다 [a, b, c, ..., blank]에 대한 확률 분포 출력.

## 최종목표

: 최종적으로 CAT가 나오는 모든 path 확률을 더하기.
`C _ A T`, `_ C A _ T`, `C C A T T` 등등...

## forward algorithm

:$$\alpha_t(s) = \left( \underbrace{\alpha_{t-1}(s)}_{\text{유지}} + \underbrace{\alpha_{t-1}(s-1)}_{\text{이전}} + \underbrace{\delta \cdot \alpha_{t-1}(s-2)}_{\text{건너뛰기 (조건부)}} \right) \times \underbrace{y_{\mathbf{l}'_s}^t}_{\text{현재 확률}}$$
이걸 재귀적으로 반복.
