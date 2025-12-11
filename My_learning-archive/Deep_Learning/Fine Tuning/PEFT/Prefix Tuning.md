### 1. Prefix Tuning의 핵심 개념 및 목표

- **목표:** 대규모 사전 학습 모델(LLM 등)의 방대한 가중치($W$)를 **고정(Freeze)**한 채, 목표 작업(Task)을 수행하는 데 필요한 **극히 적은 수의 파라미터(Prefix)**만 학습시켜 전이 학습(Transfer Learning)을 수행합니다.
    
- **아이디어:** 모델의 Attention 메커니즘을 '해킹'하여, 작업에 특화된 정보를 **매 레이어마다** 주입합니다.


### 2. 작동 메커니즘 (행렬 관점)



#### Prefix Tuning 연산 (Attention with Prefix)

- **Prefix 정의:** $L$개의 토큰 길이를 가지는 **학습 가능한 Prefix 텐서** $P_{\text{tune}} \in \mathbb{R}^{L \times d_{model}}$ 를 정의합니다.
    
- $P_{\text{tune}}$을 Key와 Value 행렬에 결합합니다.
    
    $$K' = \text{Concat}(K_{\text{prefix}}, K_{\text{input}}) \in \mathbb{R}^{(L+T) \times d_{model}}$$
    
    $$V' = \text{Concat}(V_{\text{prefix}}, V_{\text{input}}) \in \mathbb{R}^{(L+T) \times d_{model}}$$
    
- **Query는 그대로:** Query($Q$)는 입력 시퀀스 $X$를 그대로 사용합니다.
    
- **Attention 재계산:** Query는 이제 $L+T$ 길이의 Key와 Value를 사용하여 Attention을 계산합니다.
    
    $$\text{Attention}(Q, K', V') = \text{Softmax}\left(\frac{Q K'^\top}{\sqrt{d_k}}\right) V'$$
    

**핵심:** 오직 **$K_{\text{prefix}}$와 $V_{\text{prefix}}$**에 해당하는 벡터만 학습되고, 나머지 $K_{\text{input}}, V_{\text{input}}$를 만드는 원래 Transformer의 가중치($W_K, W_V$)는 **고정됩니다.**

### 3. Prefix Tuning이 작동하는 이유 (Soft Prompt)

Prefix Tuning은 추가된 $L$개의 벡터 $P_{\text{tune}}$가 일종의 **Soft Prompt (부드러운 명령어)** 역할을 하여 Attention 메커니즘을 **작업에 최적화된 상태로 유도**하기 때문에 작동합니다.

- **가이드 역할:** 입력 시퀀스 $X$의 모든 토큰($Q$)은 계산 과정에서 $P_{\text{tune}}$에 접근하여, 마치 **"이 작업은 요약(Summarization) 작업이야"**라는 복잡하고 고차원적인 지침을 전달받습니다.
    
- **정보 인코딩:** 학습된 Prefix 벡터에는 태스크에 필요한 모든 문맥적 정보가 압축되어 저장됩니다.