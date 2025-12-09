**Linear Attention**은 기존 트랜스포머의 가장 큰 병목 현상인 **$O(L^2)$(제곱)의 복잡도**를 **$O(L)$(선형) 복잡도**로 줄이기 위해 고안된 혁신적인 기법입니다.

핵심은 **"행렬 곱셈의 순서를 바꾸는 것"**이며, 이를 위해 **Softmax를 커널 함수(Kernel Function)로 대체**하는 수학적 트릭을 사용합니다. 이 과정을 단계별로 상세히 설명해 드리겠습니다.

---

### 1. 문제의 발단: 표준 어텐션의 병목 현상

표준 Self-Attention 수식은 다음과 같습니다. ($L$: 문장 길이, $d$: 차원)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

1. **$Q \times K^T$ 연산:**
    
    - $(L \times d) \times (d \times L) \rightarrow \mathbf{(L \times L)}$ 크기의 거대한 어텐션 맵이 만들어집니다.
        
    - 만약 문장 길이 $L=10,000$이라면, $10,000 \times 10,000 = 1$억 개의 셀을 가진 행렬이 생깁니다. 메모리와 계산량이 폭발합니다.
        
2. **Softmax 및 $V$ 곱셈:**
    
    - 이 거대한 $(L \times L)$ 행렬에 다시 $(L \times d)$ 크기의 $V$를 곱합니다.
        

이 **가운데 $L \times L$ 행렬을 만드는 과정** 때문에 긴 문장 처리가 불가능해집니다.


--- 

### 2. 해결의 열쇠: 결합 법칙 (Associative Property)

행렬 곱셈에는 **결합 법칙**이 성립합니다. 즉, 괄호의 위치를 바꿔도 결과는 같습니다.

$$(A \times B) \times C = A \times (B \times C)$$

우리는 이 법칙을 이용해 계산 순서를 바꾸고 싶습니다.

- **기존:** $(Q \times K^T) \times V$ $\rightarrow$ $L \times L$ 생성 (느림)
    
- **목표:** $Q \times (K^T \times V)$ $\rightarrow$ $K^T(d \times L)$와 $V(L \times d)$를 먼저 곱하면 **$d \times d$** 행렬이 됩니다. $d$는 보통 작으므로(예: 64, 128), $L$이 아무리 커져도 계산량이 늘지 않습니다.
    

하지만, 'Softmax'가 문제입니다.

Softmax 함수는 비선형 함수($e^x$)이므로 행렬 곱셈 사이에 끼어 있으면 결합 법칙을 적용할 수 없습니다.
$$\text{softmax}(QK^T)V \neq Q(\text{softmax}(K^T)V)$$

---

### 3. Linear Attention의 핵심: 커널 트릭 (Kernel Trick)

Softmax를 제거하고 결합 법칙을 쓰기 위해, 두 벡터의 유사도(Similarity)를 **커널 함수 $\phi(\cdot)$의 내적**으로 정의합니다.

#### ① 유사도 함수 재정의

기존의 $\text{softmax}(q_i k_j^T)$ 항은 $e^{q_i \cdot k_j}$ 형태입니다. 이를 다음과 같은 형태로 근사(Approximation)합니다.

$$\text{Sim}(q_i, k_j) = \phi(q_i)^T \phi(k_j)$$

(여기서 $\phi(x)$는 $x$를 양수로 만드는 함수, 예: $\text{elu}(x)+1$ 또는 ReLU 등)

#### ② 수식 변환 (순서 바꾸기)

이제 Softmax라는 감옥이 사라졌으므로, 수식을 다시 쓸 수 있습니다. (정규화 항은 설명의 단순화를 위해 분자만 표시합니다.)

$$\text{Linear Attention}(Q, K, V) = \phi(Q) \left( \phi(K)^T V \right)$$

이 수식의 연산 순서는 다음과 같이 바뀝니다.

1. **Step 1: $\phi(K)^T V$ 계산**
    
    - $(d \times L) \times (L \times d) \rightarrow \mathbf{(d \times d)}$ 행렬 생성.
        
    - 이것은 문장 길이 $L$을 한 번만 훑으면($O(L)$) 만들어지는 고정 크기의 요약 행렬입니다.
        
2. **Step 2: $\phi(Q)$와 곱하기**
    
    - $(L \times d) \times (d \times d) \rightarrow (L \times d)$
        
    - 각 쿼리 토큰마다 $d \times d$ 행렬만 곱하면 끝납니다.
        

---

### 4. 복잡도 비교 (Complexity Comparison)

|**구분**|**표준 Attention**|**Linear Attention**|**차이점**|
|---|---|---|---|
|**시간 복잡도**|$O(L^2 d)$|$O(L d^2)$|$L$이 $d$보다 클 때 압도적으로 유리|
|**메모리 복잡도**|$O(L^2)$|$O(L d)$|$L \times L$ 행렬을 저장할 필요 없음|
|**병목 지점**|문장 길이 ($L$)|임베딩 차원 ($d$)|긴 시퀀스 처리에 적합|

---

### 5. 한계점 및 고려사항

"이렇게 좋으면 왜 다 이걸로 안 바꾸나요?"

1. **성능 저하:** Softmax는 아주 큰 값은 더 크게, 작은 값은 0에 가깝게 만들어 **중요한 정보에 집중(Focusing)**하는 능력이 탁월합니다. Linear Attention의 커널 근사는 분포가 평평해지는 경향이 있어, 정확도가 다소 떨어질 수 있습니다.
    
2. **훈련 불안정:** $\phi(\cdot)$ 함수 선택에 따라 학습이 불안정해질 수 있습니다.
    
3. **Causal Masking의 어려움:** Decoder-only 모델에서 미래를 가리는 Causal Masking을 적용하려면, 순차적으로 계산해야 하므로 병렬화 이점이 줄어들고 구현이 복잡해질 수 있습니다. (RNN처럼 동작하게 됨)

   -> 원래 QK^T먼저해서 L x L을 먼저 만드는 경우에는 상삼각행렬 지우기만하면 됐는데, 
   kV먼저해서 dxd 행렬 돼버리니까 순서를 한 번에 처리할 방법이 없다.
   