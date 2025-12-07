# Padding Mask
## 시작점: 자연어 시퀀스의 근본적인 문제

자연어 문장은 **길이가 제각각**이에요.

- “Hi” → 길이 2
    
- “I love deep learning” → 길이 4
    
- “이 모델은 패딩에 대해 물어보는 사용자의 질문에 상당히 길게 답변한다” → 훨씬 김
    

그런데 GPU에서 연산을 할 때는:

- 보통 `batch_size x seq_len x hidden_dim` 같은 **직사각형 텐서**를 한 번에 처리.
    
- 텐서의 두 번째 차원(시퀀스 길이)이 **모든 샘플에서 같아야** 함.
    

> 즉, **수학적으로도, 구현적으로도 “변수 길이 텐서”를 한 번에 처리하기가 까다롭기 때문에**  
> 길이를 강제로 맞추는 장치가 필요 → 그게 `padding token`.

---
## 들어가는 위치

[Embedding + PE]
      ↓
✅ Encoder Self-Attention  ← Padding Mask
      ↓
[Add & Norm]
      ↓
[FFN]
      ↓
✅ Decoder Masked Self-Attention ← Padding Mask + Causal Mask
      ↓
✅ Decoder Cross-Attention ← Padding Mask (Encoder 쪽)
      ↓
[FFN]

## 1️⃣ Encoder – Multi-Head Self-Attention

### 어떻게 적용?

- **Key 위치의 PAD 를 가림

Scores = QKᵀ / √d_k
Scores += PaddingMask   ← 여기
Attention = softmax(Scores)

---

## 2️⃣ Decoder – Masked Self-Attention


- 목적: **Encoder와 동일**
    
- 짧은 문장의 PAD 무시

Scores += PaddingMask
Scores += CausalMask (얘도 같이씀.)

## 3️⃣ Decoder – Cross Attention

- Query: **Decoder 토큰**
    
- Key / Value: **Encoder 출력**
    

👉 따라서 **Encoder 쪽 Padding Mask** 사용

Scores = QKᵀ / √d_k
Scores += EncoderPaddingMask   ← 여기
Attention = softmax(Scores)

---
# Causal Mask

## 시작점: 정보 유출 문제

문장: `나는 밥을 먹었다`

mask 없이 self-attention을 하면:

- "나는" 위치에서 "먹었다"의 embedding을 **이미 보고 있음**

> **정답을 미리 보고 문제를 푸는 꼴 (data leakage)**

loss는 잘 줄어드는데,  
실제로 생성(inference)할 땐 미래가 없어서 모델이 무너짐.

## Causal Mask의 정확한 역할

> **“t번째 토큰은, t 이전까지만 볼 수 있다”**

이를 attention score 차원에서 강제함. ( X : 차단 )

 i/j   1    2   3   4   5
-----------------------
1     O   X   X   X   X
2     O   O   X   X   X
3     O   O   O   X   X
4     O   O   O   O   X
5     O   O   O   O   O

Self attention 단계에서 QK^T 를 내적하고 softmax에 넣기 직전에 + M (masking)을 함.

- M에는 통상적으로 음의 무한대 값을 넣어서, softmax에 들어갔을 때 0으로 수렴해버리게 함.

최종 X_att = softmax(QK^T / root(d_k) + M) * V




# ✅ Encoder/Decoder 모두에서 mask 일관성 유지


### 학습시:
- **Encoder Self-Attention:**
    
    - Padding Mask: **YES** (PAD 무시)
        
    - Causal Mask: **NO** (번역할 때 문장 전체를 다 봐야 문맥을 아니까요.)
        
- **Decoder Self-Attention:**
    
    - Padding Mask: **YES**
        
    - Causal Mask: **YES** (생성할 때 미래를 보면 안 됨.) -> **보통 이 둘을 합쳐서 하나의 마스크로 만듭니다.**
        
- **Cross-Attention (Decoder가 Encoder를 볼 때):**
    
    - Query(Decoder), Key/Value(Encoder)
        
    - 여기서는 **Encoder 쪽의 Padding Mask**를 적용해야 합니다. (Decoder가 Encoder의 PAD를 보지 않도록)
        
    - Causal Mask는 적용하지 않습니다. (이미 완성된 Encoder의 문장은 다 봐도 되니까요.)

### 추론시:
