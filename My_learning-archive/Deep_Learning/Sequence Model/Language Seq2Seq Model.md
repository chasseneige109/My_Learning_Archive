
# LSTM 기반 언어 모델(Next Word Prediction) — A to Z (Obsidian Friendly)

---

## 0. 기본 설정 (Setup)

### 파라미터 크기

- 단어장 크기( Vocabulary Size ):  
    V=10,000
    
- 임베딩 차원(Embedding Size):  
    D=128
    
- LSTM 히든 차원(Hidden Size):  
    H=256
    

### 학습해야 할 행렬들

1. **임베딩 행렬**
    
    E∈RV×D=10000×128\mathbf{E} \in \mathbb{R}^{V \times D} = 10000 \times 128E∈RV×D=10000×128
2. **LSTM 가중치**  
    4개 게이트를 한 번에 처리하기 위해 크기:
    
    - 입력 가중치:
        
        Wx∈RD×4H=128×1024\mathbf{W_x} \in \mathbb{R}^{D \times 4H} = 128 \times 1024Wx​∈RD×4H=128×1024
    - 히든 가중치:
        
        Wh∈RH×4H=256×1024\mathbf{W_h} \in \mathbb{R}^{H \times 4H} = 256 \times 1024Wh​∈RH×4H=256×1024
    - 편향:
        
        b∈R1×1024\mathbf{b} \in \mathbb{R}^{1 \times 1024}b∈R1×1024
3. **출력 투영 행렬**
    
    Wproj∈RH×V=256×10000\mathbf{W}_{proj} \in \mathbb{R}^{H \times V} = 256 \times 10000Wproj​∈RH×V=256×10000
4. **출력 편향**
    
    bproj∈R1×10000\mathbf{b}_{proj} \in \mathbb{R}^{1 \times 10000}bproj​∈R1×10000

---

# Phase 1 — 학습 (Training): Forward Pass

입력 시퀀스:

`<SOS>, I, love, AI`

정답 시퀀스:

`I, love, AI, <EOS>`

예로 시간 t에서 입력 단어가 `"love"`라고 하자(단어 인덱스 450).

---

## Step 1. 임베딩 조회 (Embedding Lookup)

원-핫 벡터(1×10000)를 쓰지 않고, 그냥 인덱스를 이용해 한 행을 뽑는다.

xt=E[450]\mathbf{x}_t = \mathbf{E}[450]xt​=E[450]

크기:

1×1281 \times 1281×128

---

## Step 2. LSTM 연산

### 1) 선형 결합

입력과 이전 히든 상태를 이용하여 4H 크기의 벡터 생성:

z=xtWx+ht−1Wh+b\mathbf{z} = \mathbf{x}_t \mathbf{W_x} + \mathbf{h}_{t-1} \mathbf{W_h} + \mathbf{b}z=xt​Wx​+ht−1​Wh​+b

크기:

1×10241 \times 10241×1024

---

### 2) 게이트 분할 및 활성화

z=[zf  ∣  zi  ∣  zg  ∣  zo]\mathbf{z} = [\mathbf{z}_f \;|\; \mathbf{z}_i \;|\; \mathbf{z}_g \;|\; \mathbf{z}_o]z=[zf​∣zi​∣zg​∣zo​]

각각 256차원씩.

- forget gate:
    
    ft=σ(zf)\mathbf{f}_t = \sigma(\mathbf{z}_f)ft​=σ(zf​)
- input gate:
    
    it=σ(zi)\mathbf{i}_t = \sigma(\mathbf{z}_i)it​=σ(zi​)
- candidate content:
    
    C~t=tanh⁡(zg)\tilde{\mathbf{C}}_t = \tanh(\mathbf{z}_g)C~t​=tanh(zg​)
- output gate:
    
    ot=σ(zo)\mathbf{o}_t = \sigma(\mathbf{z}_o)ot​=σ(zo​)

---

### 3) 셀/히든 상태 업데이트

셀 업데이트:

Ct=ft⊙Ct−1+it⊙C~t\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_tCt​=ft​⊙Ct−1​+it​⊙C~t​

히든 상태:

ht=ot⊙tanh⁡(Ct)\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)ht​=ot​⊙tanh(Ct​)

두 벡터 모두 크기:

1×2561 \times 2561×256

---

## Step 3. 출력 Projection → Logits

logitst=htWproj+bproj\mathbf{logits}_t = \mathbf{h}_t \mathbf{W}_{proj} + \mathbf{b}_{proj}logitst​=ht​Wproj​+bproj​

크기:

1×100001 \times 100001×10000

---

## Step 4. Softmax → 단어 확률

Pt=Softmax(logitst)\mathbf{P}_t = \text{Softmax}(\mathbf{logits}_t)Pt​=Softmax(logitst​)

단어 1만 개에 대한 확률 분포.

---

## Step 5. Loss 계산

정답 단어 `"AI"`의 인덱스가 900이라고 하자.

Losst=−log⁡(Pt,900)\text{Loss}_t = -\log(\mathbf{P}_{t, 900})Losst​=−log(Pt,900​)

---

# Phase 2 — Backward Pass (BPTT)

Softmax Cross Entropy의 미분은 다음과 같다:

∂L∂logitst=Pt−yt\frac{\partial L}{\partial \mathbf{logits}_t} = \mathbf{P}_t - \mathbf{y}_t∂logitst​∂L​=Pt​−yt​

여기서 $\mathbf{y}_t$는 원-핫 벡터.

오차는 다음 순서로 전달되어 가중치들이 업데이트됨:

1. $\mathbf{W}_{proj}$
    
2. LSTM 내부 가중치 $\mathbf{W_x}, \mathbf{W_h}, \mathbf{b}$
    
3. 임베딩 행렬 $\mathbf{E}$의 450번째 행
    

최종 업데이트:

W←W−α∇W\mathbf{W} \leftarrow \mathbf{W} - \alpha \nabla \mathbf{W}W←W−α∇W

---

# Phase 3 — 추론(Inference): 문장 생성 (Autoregressive)

시작 입력은 항상 `<SOS>`.

## Time 1

1. 입력: `<SOS>`
    
2. LSTM → $\mathbf{h}_1$
    
3. Softmax → 다음 단어 확률
    
4. argmax 또는 sampling → `"I"`
    

출력 단어: `"I"`

---

## Time 2 — **Autoregression 핵심**

이번 입력은 정답이 아니라 **방금 모델이 생성한 단어 `"I"`**.

1. 입력: `"I"`
    
2. LSTM → $\mathbf{h}_2$
    
3. Softmax → `"love"`
    

---

## Time 3

1. 입력: `"love"`
    
2. LSTM → $\mathbf{h}_3$
    
3. Softmax → `"AI"`
    

---

## Time 4

1. 입력: `"AI"`
    
2. Softmax → `<EOS>`
    
3. `<EOS>`가 나왔으므로 종료
    

---

# 최종 생성 결과

`I love AI`

---

# 전체 과정 요약 (Obsidian 호환)

1. 정수 인덱스 → 임베딩 조회
    
2. 임베딩 + 이전 히든 → LSTM(4게이트 연산)
    
3. 새로운 히든 상태 → Projection → Softmax
    
4. Cross Entropy로 Loss 계산
    
5. BPTT로 가중치 업데이트
    
6. 추론에서는 이전에 생성한 단어를 다음 단계 입력으로 사용