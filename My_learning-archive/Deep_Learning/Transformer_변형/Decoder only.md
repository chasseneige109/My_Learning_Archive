

**자기회귀(Autoregressive) 학습** / **티처 포싱(Teacher Forcing)**

**Shifted target**

| **위치 (Index)**          | **1**              | **2**              | **3**               | **4**              | **5**             | **6**                 |
| ----------------------- | ------------------ | ------------------ | ------------------- | ------------------ | ----------------- | --------------------- |
| **입력 시퀀스 (X, Context)** | `[CLS]`            | `나는`               | `어제`                | `마트에`              | `갔다`              | `.`                   |
| **모델의 예측**              | $\rightarrow$ `나는` | $\rightarrow$ `어제` | $\rightarrow$ `마트에` | $\rightarrow$ `갔다` | $\rightarrow$ `.` | $\rightarrow$ `[SEP]` |
| **정답 시퀀스 (Y, Target)**  | `나는`               | `어제`               | `마트에`               | `갔다`               | `.`               | `[SEP]`               |
|                         |                    |                    |                     |                    |                   |                       |

### **1단계: 입력 구성 (Input Embedding)**

GPT 같은 Decoder 모델은 보통 문장 구분(Segment Embedding) 없이 **토큰**과 **위치**만 사용합니다.

- **입력:** `["Robot", "moves"]` (길이 $L=2$)
    
- **사전 크기:** $|V|$ (예: 50,000)
    
- **임베딩 차원:** $d$ (예: 768)
    

1. **Lookup Table:**
    
    - $W_T$ (토큰 임베딩): `Robot`의 ID와 `moves`의 ID에 해당하는 벡터를 가져옵니다.
        
    - $W_P$ (포지션 임베딩): 위치 0, 위치 1에 해당하는 벡터를 가져옵니다.
        
2. Element-wise Sum:
    
    $$X_0 = \text{Lookup}(W_T) + \text{Lookup}(W_P)$$
    
    - **결과 행렬 $X_0$:** 크기는 **$[L \times d]$** (즉, $2 \times 768$)입니다.
        

---

### **2단계: 마스크드 셀프 어텐션 (Masked Self-Attention) - 핵심**

이곳이 Decoder의 심장입니다. $X$ 행렬이 들어와서 어텐션을 수행합니다.

#### 1. Q, K, V 투영 (Linear Projection)

입력 행렬 $X$에 가중치 행렬 $W_Q, W_K, W_V$를 곱해 3개의 행렬을 만듭니다.

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

- 모두 크기는 $[L \times d]$입니다.
    

#### 2. 어텐션 스코어 계산 ($Q \times K^T$)

쿼리와 키를 내적하여 유사도를 구합니다.

$$\text{Score} = \frac{Q K^T}{\sqrt{d_k}}$$

- **결과 행렬 크기:** $[L \times L]$ ($2 \times 2$)
    
- 이 행렬은 "누가 누구를 얼마나 쳐다볼지"를 나타냅니다.
    

#### 3. **Causal Masking (Look-ahead Mask)**

이것이 Encoder와의 결정적 차이입니다.

자기보다 미래에 있는 단어는 볼 수 없도록 마스킹합니다. 상삼각 행렬(Upper Triangular Matrix) 부분을 $-\infty$로 채웁니다.

- **마스크 행렬:**
    
    $$ M = \begin{bmatrix} 0 & -\infty \\ 0 & 0 \end{bmatrix}$$
    
- **의미:**
    
    - 1행 (Robot): 자기 자신(`Robot`)은 볼 수 있지만(0), 미래인 `moves`는 못 봅니다($-\infty$).
        
    - 2행 (moves): 과거(`Robot`)와 현재(`moves`)를 모두 볼 수 있습니다(0).
        

#### 4. Softmax 및 Value 결합

$$Z = \text{Softmax}(\text{Score} + M) \times V$$

- $-\infty$였던 부분은 Softmax를 거치면 **0**이 됩니다. 즉, 미래 정보가 완벽히 차단됩니다.
    
- **결과:** $Z$ 행렬 $[L \times d]$ 생성.
    

---

### **3단계: Feed-Forward Network (FFN)**

어텐션이 "단어 간의 관계"를 봤다면, FFN은 "단어 자체의 의미"를 심화 학습합니다. 각 토큰(행)별로 독립적으로 계산됩니다.

$$H = \text{GELU}(Z W_1 + b_1) W_2 + b_2$$

1. **확장 ($W_1$):** 차원을 4배로 뻥튀기합니다 ($d \rightarrow 4d$). (예: $768 \rightarrow 3072$)
    
2. **활성화 함수:** ReLU나 GELU를 통과시켜 비선형성을 추가합니다.
    
3. **축소 ($W_2$):** 다시 원래 차원으로 줄입니다 ($4d \rightarrow d$).
    

---

### **4단계: 잔차 연결 및 정규화 (Add & Norm)**

각 블록(Attention, FFN)이 끝날 때마다 잊지 말아야 할 두 가지가 있습니다.

1. **Residual Connection (잔차 연결):** 입력과 출력을 더합니다. ($X + \text{Layer}(X)$). 그래야 정보 손실 없이 깊게 쌓을 수 있습니다.
    
2. **Layer Normalization:** 데이터 분포를 깔끔하게 정리합니다.
    

이 **[Attention $\rightarrow$ Add&Norm $\rightarrow$ FFN $\rightarrow$ Add&Norm]** 블록을 $N$번(예: 12~96번) 반복합니다.

---

### **5단계: 출력층 (Output Head & Unembedding)**

마지막 블록을 통과한 최종 행렬 $H_{\text{final}}$ ($L \times d$)이 나왔습니다. 이제 다음 단어를 예측해야 합니다.

1. Unembedding (Linear Projection):
    
    임베딩 차원($d$)을 다시 단어 집합 크기($|V|$)로 변환하는 행렬 $W_U$를 곱합니다.
    
    $$\text{Logits} = H_{\text{final}} \times W_U$$
    
    - **결과 크기:** $[L \times |V|]$ (즉, $2 \times 50,000$)
        
2. 마지막 토큰만 주목 (Next Token Prediction):
    
    우리는 Robot moves 다음에 올 단어가 궁금합니다. 따라서 행렬의 **마지막 행(Last Row)**만 떼어냅니다.
    
    - 마지막 행 벡터: $[1 \times 50,000]$
        
3. Softmax:
    
    이 벡터를 Softmax에 통과시켜 50,000개 단어에 대한 확률 분포를 얻습니다.
    
    - 예: `fast`(20%), `slowly`(15%), `forward`(10%)...
        

---

### **6단계: 디코딩 및 루프 (Inference Loop)**

이제 확률이 가장 높은 단어(예: `fast`)를 선택(Sampling)합니다. 여기서 끝이 아닙니다.

1. **Append (이어 붙이기):** 생성된 단어 `fast`를 기존 입력에 붙입니다.
    
    - 새 입력: `["Robot", "moves", "fast"]`
        
2. **반복 (Loop):** 이 새로운 입력을 가지고 **1단계부터 다시 시작**합니다.
    

이 과정을 `[EOS]`(문장 종료 토큰)이 나올 때까지 반복합니다.