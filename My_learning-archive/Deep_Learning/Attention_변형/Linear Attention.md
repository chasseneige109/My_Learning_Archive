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