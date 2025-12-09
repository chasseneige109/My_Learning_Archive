**FlashAttention**은 2022년 스탠포드 연구진이 발표한 논문("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")에서 제안된 기술로, 현대 LLM 학습과 추론 속도를 획기적으로 높인 일등 공신입니다.

핵심은 **"연산 횟수(FLOPs)를 줄이는 게 아니라, 메모리 이동(Memory Access)을 줄여서 속도를 높인다"**는 것입니다. 이를 **IO-Aware(입출력 인지)** 알고리즘이라고 합니다.

상세하게 구조와 원리를 분해해 드리겠습니다.

---

### 1. 배경: GPU 메모리 계층 구조의 이해 (Warehouse vs Desk)

GPU 성능을 이해하려면 두 가지 메모리를 알아야 합니다.

1. **HBM (High Bandwidth Memory):**
    
    - **비유:** 거대한 **물류 창고**.
        
    - **특징:** 용량이 매우 큼(40GB~80GB), 하지만 데이터를 가져오는 속도가 상대적으로 느림.
        
    - **역할:** 모델의 파라미터($Q, K, V$ 전체)와 전체 데이터가 저장됨.
        
2. **SRAM (Static RAM, On-chip Memory):**
    
    - **비유:** 작업자의 **작은 책상**.
        
    - **특징:** 용량이 매우 작음(A100 기준 19MB 정도), 하지만 계산기(Compute Unit) 바로 옆이라 **엄청나게 빠름**.
        
    - **역할:** 실제 연산(행렬 곱 등)은 여기서 일어남.
        

**문제점:** GPU 연산 속도는 엄청 빠른데, **창고(HBM)에서 책상(SRAM)으로 데이터를 나르는 시간**이 너무 오래 걸립니다. (Memory Bandwidth Bottleneck)

---

### 2. 기존 Attention (Standard Attention)의 병목

기존 방식은 $N \times N$ 크기의 거대한 어텐션 행렬을 HBM에 쓰고 읽는 과정을 반복합니다.

$$S = QK^T, \quad P = \text{Softmax}(S), \quad O = PV$$

1. HBM에서 $Q, K$를 SRAM으로 가져와 $S = QK^T$ 계산 $\rightarrow$ **결과 $S$를 HBM에 저장** ($N \times N$ 크기, 엄청 큼).
    
2. HBM에서 $S$를 다시 가져와 `Softmax` 계산 $\rightarrow$ **결과 $P$를 HBM에 저장**.
    
3. HBM에서 $P$와 $V$를 가져와 곱셈 $\rightarrow$ **최종 결과 $O$를 HBM에 저장**.
    

**결론:** 거대한 $N \times N$ 행렬을 HBM에 썼다가 읽었다가 하는 과정이 너무 느립니다.

---

### 3. FlashAttention의 핵심 아이디어

FlashAttention의 목표는 **"중간 결과($S, P$)를 HBM에 기록하지 말고, SRAM 안에서 끝내버리자"**입니다. 이를 위해 두 가지 핵심 기술을 사용합니다.

#### ① 타일링 (Tiling) - "조각내서 처리하기"

$Q, K, V$ 행렬 전체를 한 번에 처리하는 대신, SRAM 용량에 맞는 **작은 블록(Tile)** 단위로 쪼갭니다.

- $Q$를 $B_r$ 크기의 블록들로 나눕니다.
    
- $K, V$를 $B_c$ 크기의 블록들로 나눕니다.
    
- 작은 블록들을 SRAM에 올린 뒤, **SRAM 안에서** `MatMul` $\rightarrow$ `Softmax` $\rightarrow$ `MatMul`을 모두 수행합니다.
    
- 이러면 거대한 $N \times N$ 행렬을 HBM에 기록할 필요가 없습니다. (Kernel Fusion)
    

#### ② 온라인 소프트맥스 (Online Softmax) - "수학적 트릭"

문제는 **Softmax**입니다. Softmax를 계산하려면 **전체 행의 최대값(Max)과 합(Sum)**을 알아야 분모(정규화 상수)를 구할 수 있습니다. 블록 단위로 쪼개면 전체 값을 모르는 상태에서 계산해야 합니다.

이를 해결하기 위해 **Online Softmax** 알고리즘을 사용합니다.

- **아이디어:** 데이터를 순차적으로 보면서, 새로운 최대값이 나타날 때마다 **이전까지 계산한 결과를 보정(Rescaling)**하는 방식입니다.
    

수식적 원리:

두 블록 $x_1$과 $x_2$가 있다고 가정합시다.

1. **블록 1 처리:** 국소적 최대값 $m_1$을 이용해 임시 Softmax 값 계산.
    
2. **블록 2 처리:** 국소적 최대값 $m_2$ 발견.
    
3. **통합 및 보정:** 전체 최대값 $m_{new} = \max(m_1, m_2)$를 구한 뒤, 블록 1의 결과에 $e^{m_1 - m_{new}}$만큼 곱해서 값을 보정해 줍니다.
    

$$O_{new} = \text{diag}(e^{m_{old} - m_{new}}) \times O_{old} + e^{S_{new} - m_{new}} \times V_{new}$$

이렇게 하면 전체 데이터를 한 번에 보지 않고도, **블록 단위로 순회하면서 정확한 Softmax 결과**를 만들어낼 수 있습니다.

---

### 4. FlashAttention 동작 과정 (Forward Pass)

1. **Load:** HBM에서 $Q$의 블록($Q_i$), $K, V$의 블록($K_j, V_j$)을 SRAM으로 가져옵니다.
    
2. **Compute:** SRAM 내에서 점곱($Q_i K_j^T$)을 수행합니다.
    
3. **Online Softmax:**
    
    - 현재 블록의 `max`와 `sum`을 계산합니다.
        
    - 이전 블록까지의 결과($O_i$)를 새로운 `max`에 맞춰 스케일링(Rescaling)하고 현재 결과를 더합니다.
        
4. **Write:** 최종적으로 업데이트된 결과 $O_i$만 HBM에 기록합니다. (중간 행렬 $S, P$는 HBM에 안 씀)
    

---

### 5. 또 하나의 비밀: Backward Pass (Recomputation)

학습(Backpropagation)을 하려면 Forward 때 계산했던 $N \times N$ Attention 행렬값들이 필요합니다. 기존 방식은 이걸 HBM에 저장해 뒀다가 꺼내 썼습니다.

- **FlashAttention의 역발상:** "저장하지 말고 그냥 **다시 계산(Recomputation)**하자."
    
- HBM에 저장하고 읽어오는 속도보다, SRAM에서 GPU 코어로 다시 계산하는 게 **더 빠르기 때문**입니다.
    
- 이를 통해 **메모리 사용량을 $O(N^2)$에서 $O(N)$으로** 획기적으로 줄였습니다.
    

---

### 6. 요약 및 효과

|**구분**|**Standard Attention**|**FlashAttention**|
|---|---|---|
|**HBM 접근**|$O(N^2)$ (입출력 많음)|**$O(N)$ (선형적)**|
|**메모리 사용량**|$O(N^2)$ (거대 행렬 저장)|**$O(N)$ (선형적)**|
|**속도**|HBM 대역폭에 의해 제한됨|**2배~4배 더 빠름**|
|**긴 문장 처리**|메모리 부족(OOM)으로 어려움|**매우 긴 문맥(16k, 32k...) 가능**|

**결론적으로 FlashAttention은:**

1. 입출력(IO)이 병목이라는 점을 간파하고
    
2. **Tiling**을 통해 연산을 SRAM 안으로 숨겼으며
    
3. **Online Softmax**라는 수학적 기법으로 이를 가능하게 만들었습니다.
    

이 기술 덕분에 GPT-4 같은 모델이 수만 토큰의 문맥을 한 번에 처리할 수 있게 된 것입니다.