
### Scheduled Sampling (Bengio et al., 2015)

훈련 중 매 step마다:

- 확률 p: 정답 토큰 사용
- 확률 1−p: 모델 예측 토큰 사용
- 
단점:
- **미분불가