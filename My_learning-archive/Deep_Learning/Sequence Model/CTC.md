# 목표: 
음성 인식, 필기 인식: **어느 시점에** 문자가 나와야 하는지 알 수 없음.
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
   (확률 행렬은 미리 계산되어 받아놓은 상태)
    
3. **시작:** $t=1$일 때 첫 빈칸과 첫 글자 칸에만 초기 로그 확률 주입.
    
4. **루프:** $t=2 \to T$ 동안
    
    - 각 상태 $s$에 대해 **유지/진입/점프** 가능성을 체크.
        
    - 가능한 전 단계의 확률들을 **LogSumExp**로 합침.
        
    - 현재 시점 $t$의 해당 문자 로그 확률을 **더함(곱셈)**.
        
5. **결과:** 마지막 시간 $T$의 '마지막 빈칸'과 '마지막 글자'의 확률을 합침.
    


## backward algorithm

forward와 기본적으로 예쁘게 대칭적임.
Forward의 인덱스 방향만 반대

$$\beta_t(s) = \left( \underbrace{\beta_{t+1}(s)}_{\text{유지}} + \underbrace{\beta_{t+1}(s+1)}_{\text{다음}} + \underbrace{\delta \cdot \beta_{t+1}(s+2)}_{\text{점프}} \right) \times \underbrace{y_{\mathbf{l}'_s}^t}_{\text{현재 확률}}$$
T에서 1로 거슬러 올라가면서 계산하기 때문에,
(t+1)에서의 $beta$ 값들은 전부 이미 계산되어있음.
forward에서 (t-1)값들이 전부 이미 계산되어있는 것처럼.


## forward, backward 종합하기
$$P(\pi_t = s | \mathbf{x}, \mathbf{l}) = \gamma_t(s) = \frac{\alpha_t(s) \cdot \beta_t(s)}{y_{\mathbf{l}'_s}^t \cdot P(\mathbf{l}|\mathbf{x})}$$y가 a,b에 중복해서 곱해서있어서 1번 나눔.

* 감마는 CAT을 출력하는것에 성공할경우, t에서 상태s일 확률

$$\gamma_t(s) = \frac{\alpha_t(s) \cdot \beta_t(s) / y_{\mathbf{l}'_s}^t}{\alpha_T(|\mathbf{l}'|) + \alpha_T(|\mathbf{l}'|-1)}$$ 이렇게도 표현가능


## backpropagation

이건 뭐.. 걍 해라

## gradient descent

$$\frac{\partial \mathcal{L}}{\partial u_k^t} = y_k^t - \sum_{s : \mathbf{l}'_s = k} \gamma_t(s)$$

- **$y_k^t$ (Prediction):** 모델이 지금 'A'라고 생각하는 확률.
    
- **$\sum \gamma$ (Target):** 전체 문맥을 고려했을 때, 여기서 'A'가 나왔어야 할 **진짜 확률**.
    
의미: (내 예측) - (실제 정답 비율)


## 학습 후 추론 및 디코딩

실전에서 추론할 때는 정답 레이블($\mathbf{l}$)을 모르기 때문에 $\beta$를 구할 수 없습니다. 그렇다고 모든 경로를 다 고려하며 가면 exponential한 시간복잡도를 가지기 때문에, 다음 방법들 중 하나를 씀.
###  1. Beam Search Decoding (빔 서치)

- **방법:** 매 시간 확률이 높은 **상위 K개(Beam Width)**의 경로를 살려두고 끝까지 가져갑니다.
    
- **특징:** 언어 모델(Language Model, 예: 단어 사전) 점수를 추가로 반영할 수 있어, 발음은 비슷하지만 문법적으로 맞는 단어를 찾아낼 수 있습니다.
    
    - 예: "I ate an **apple**" vs "I ate an **appel**" $\to$ 언어 모델이 apple을 선택.

### 2. greedy algorithm.. (don't use)



# 단점

1. 이론적 단점: 조건부 독립 가정의 문제 이론적 단점: 조건부 독립 가정의 문제 🧠
2. 실용적 단점: 복잡한 추론 과정과 비효율성 ⚙️
3. 출력 길이의 제약 (L ≤ T)
4. 구조적 단점: 빈칸 심볼의 비효율성 💨