**KV Caching (Key-Value Caching)**은 LLM(대규모 언어 모델)의 **추론(Inference) 속도**를 획기적으로 높여주는 가장 기본적이면서도 중요한 최적화 기술입니다.

특히 **오토리그레시브(Autoregressive) 디코더** 모델(GPT, Llama 등)이 텍스트를 생성할 때 발생하는 **중복 연산**을 제거하는 것이 핵심입니다. 이를 **연산 과정(Compute)**과 **메모리 구조(Memory)** 관점에서 깊이 있게 설명해 드리겠습니다.

---

### 1. 문제 상황: 왜 비효율적인가? (Without Cache)

GPT가 "Robot moves fast"라는 문장을 생성한다고 가정해 봅시다.

#### Step 1: "Robot" 입력 $\rightarrow$ "moves" 생성

- 입력: `["Robot"]`
    
- 연산: `Robot`에 대한 $Q_1, K_1, V_1$을 계산하고 어텐션을 수행합니다.
    

#### Step 2: "moves" 입력 $\rightarrow$ "fast" 생성

- 입력: `["Robot", "moves"]` (전체 문맥을 다시 넣어야 함)
    
- 연산:
    
    - **`Robot`에 대한 $K_1, V_1$을 다시 계산합니다.** (낭비!)
        
    - `moves`에 대한 $K_2, V_2$를 계산합니다.
        
    - 어텐션 수행.
        

#### Step 3: "fast" 입력 $\rightarrow$ "." 생성

- 입력: `["Robot", "moves", "fast"]`
    
- 연산:
    
    - **`Robot` ($K_1, V_1$) 다시 계산.** (낭비!)
        
    - **`moves` ($K_2, V_2$) 다시 계산.** (낭비!)
        
    - `fast` ($K_3, V_3$) 계산.
        

**문제점:** 시퀀스 길이가 $T$일 때, 매 스텝마다 $1 \sim (T-1)$까지의 토큰에 대해 행렬 투영(Projection) 연산을 반복해야 합니다. 이는 **$O(T^2)$의 연산 낭비**를 초래합니다.

---

### 2. 해결책: KV Caching의 작동 원리

**"과거는 변하지 않는다"**는 점을 이용합니다. `Robot`이라는 단어의 의미($K, V$ 벡터)는 뒤에 `moves`가 오든 `runs`가 오든 변하지 않습니다. 그러니 **저장(Cache)**해두자는 것입니다.

#### Step 1: "Robot" 처리

- 연산: $K_1, V_1$ 계산.
    
- **저장:** GPU 메모리(VRAM)의 **KV Cache** 영역에 $[K_1], [V_1]$을 저장합니다.
    
- 출력: "moves"
    

#### Step 2: "moves"만 입력 (Next Token)

- 입력: `["moves"]` (이전 전체 문장이 아님!)
    
- 연산:
    
    1. 현재 토큰 `moves`에 대한 $Q_2, K_2, V_2$만 계산합니다.
        
    2. **로드:** 캐시에서 과거 데이터 $[K_1], [V_1]$을 가져옵니다.
        
    3. **결합 (Concat):**
        
        - $K_{total} = [K_1; K_2]$
            
        - $V_{total} = [V_1; V_2]$
            
    4. **어텐션:** $Q_2$와 $K_{total}$로 어텐션 스코어를 구하고 $V_{total}$과 곱합니다.
        
- **저장:** 캐시에 $K_2, V_2$를 추가(Append)합니다.
    

**결과:** 매 스텝마다 **현재 토큰 딱 1개**에 대해서만 행렬 연산을 수행하면 됩니다.

---

### 3. 행렬 관점에서의 심층 분석

행렬의 차원(Dimension)을 보면 KV Cache의 효율성을 명확히 알 수 있습니다.

($d$: hidden dimension, $H$: Head 수 무시하고 단일 헤드 가정)

#### 캐시가 없을 때 (Step $t$)

입력 $X$의 크기는 $[t \times d]$입니다.

1. **Projection:** $X \times W_K \rightarrow K [t \times d]$ (행렬 곱셈 비용 큼)
    
2. **Attention:** $Q [t \times d] \times K^T [d \times t] \rightarrow [t \times t]$
    

#### 캐시가 있을 때 (Step $t$)

입력 $x_{new}$의 크기는 $[1 \times d]$입니다. (토큰 1개)

1. **Projection:** $x_{new} \times W_K \rightarrow k_{new} [1 \times d]$ (**연산량 1/t 로 감소**)
    
2. **Cache Append:**
    
    - $K_{cache}: [(t-1) \times d]$
        
    - $K_{total} = \text{Concat}(K_{cache}, k_{new}) \rightarrow [t \times d]$
        
3. **Attention:**
    
    - $q_{new} [1 \times d] \times K_{total}^T [d \times t] \rightarrow \text{Score} [1 \times t]$
        
    - $\text{Score} [1 \times t] \times V_{total} [t \times d] \rightarrow \text{Output} [1 \times d]$
        

**핵심:** Projection 연산(W 곱하기)을 $O(T)$에서 **$O(1)$**로 줄였습니다. (Attention 자체는 여전히 과거 모든 토큰과 비교해야 하므로 $O(T)$가 들지만, 무거운 행렬 생성 비용을 아꼈습니다.)

---

### 4. Trade-off: 메모리 사용량 (VRAM 폭발)

KV Caching은 **"연산 속도(Compute)"를 얻는 대신 "메모리(Memory)"를 희생**하는 기술입니다.

#### 메모리 용량 계산

KV Cache가 차지하는 VRAM 용량은 다음과 같습니다. (Float16 기준 2바이트)

$$\text{Size} = 2 \times 2 \times n_{layers} \times d_{model} \times n_{seq} \times n_{batch}$$

(첫 번째 2는 K와 V, 두 번째 2는 bytes)

- **문제점:** 시퀀스 길이($n_{seq}$)와 배치 크기($n_{batch}$)에 비례하여 메모리 사용량이 선형적으로 증가합니다.
    
- **현실:** 긴 문맥(Long Context, 예: 128k 토큰)을 처리하거나 동시 접속자(Batch)가 많으면, 모델 가중치보다 **KV Cache가 메모리를 더 많이 차지**하여 OOM(Out of Memory)을 유발하는 주범이 됩니다.
    

---

### 5. 심화: KV Cache 최적화 기술들

메모리 부족 문제를 해결하기 위해 최신 모델들은 변형된 어텐션을 사용합니다.

#### (1) MQA (Multi-Query Attention)

- **아이디어:** "K와 V는 굳이 헤드마다 다를 필요가 있을까?"
    
- **방법:** 모든 Head가 **하나의 K, V**를 공유합니다. (Q는 여전히 여러 개)
    
- **효과:** KV Cache 크기가 헤드 개수($H$)만큼 줄어듭니다. (예: 1/8로 감소)
    
- **단점:** 성능이 약간 저하될 수 있음.
    

#### (2) GQA (Grouped-Query Attention) - _Llama 2, 3 채택_

- **아이디어:** MQA는 너무 심하니 절충하자.
    
- **방법:** 헤드를 몇 개의 그룹으로 묶고, 그룹 내에서만 K, V를 공유합니다.
    
- **효과:** 성능은 MHA(Multi-Head Attention)와 비슷하면서, 메모리는 MQA급으로 절약합니다.
    

#### (3) PagedAttention (vLLM) - _시스템 레벨 최적화_

- **문제:** KV Cache를 위해 미리 긴 메모리를 할당해두는 것은 낭비입니다. (Fragmentation)
    
- **해결:** 운영체제(OS)의 **가상 메모리 페이징(Paging)** 기법을 도입.
    
- **방법:** KV Cache를 불연속적인 메모리 블록(Page)에 나누어 저장하고, 필요할 때마다 동적으로 할당합니다.
    
- **효과:** 메모리 낭비를 없애 배치 크기를 키울 수 있어 처리량(Throughput)이 대폭 상승합니다.
    

### 요약

1. **KV Caching:** 과거 토큰의 K, V를 VRAM에 저장해두고 재사용하는 기술.
    
2. **이득:** 매 스텝 행렬 연산 비용을 $O(T^2)$에서 $O(T)$ 수준으로 최적화 (Projection은 $O(1)$).
    
3. **비용:** 시퀀스가 길어질수록 GPU 메모리(VRAM)를 엄청나게 잡아먹음.
    
4. **해결:** GQA(모델 구조 변경)나 PagedAttention(메모리 관리 최적화) 등으로 메모리 효율을 높임.