
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
a 계산시, 계속 underflow되므로, log를 씌워서 크기 보존

### 요약: Forward 알고리즘의 흐름도

1. **확장:** 정답 `CAT` $\to$ `_ C _ A _ T _`
    
2. **준비:** $(T \times 7)$ 크기의 $\alpha$ 행렬 생성 (모두 $-\infty$로 초기화).
   확률 행렬은 미리 곗
    
3. **시작:** $t=1$일 때 첫 빈칸과 첫 글자 칸에만 초기 로그 확률 주입.
    
4. **루프:** $t=2 \to T$ 동안
    
    - 각 상태 $s$에 대해 **유지/진입/점프** 가능성을 체크.
        
    - 가능한 전 단계의 확률들을 **LogSumExp**로 합침.
        
    - 현재 시점 $t$의 해당 문자 로그 확률을 **더함(곱셈)**.
        
5. **결과:** 마지막 시간 $T$의 '마지막 빈칸'과 '마지막 글자'의 확률을 합침.
    

