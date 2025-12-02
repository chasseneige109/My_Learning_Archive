
# Connectionist Temporal Classification


CAT을 맞춘다고하자.

일단 빈칸을 사이사이에 끼워넣어 expansion함
()C()A()T()  --> 2*L + 1개로 늘어남

나중에 연속 중복 문자는 1개로 축소시킴


매 시간 T마다 [a, b, c, ..., blank]에 대한 확률 분포 출력.

## 최종목표

: 최종적으로 CAT가 나오는 모든 path 확률을 더하기.
`C _ A T`, `_ C A _ T`, `C C A T T` 등등...



