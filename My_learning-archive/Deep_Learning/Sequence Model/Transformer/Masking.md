# Padding Mask
## 1. 시작점: 자연어 시퀀스의 근본적인 문제

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

## 2. “그럼 그냥 한 문장씩 처리하면 되지 않나요?”

이론적으로는 가능해요. 하지만:

1. **GPU 병렬처리 효율이 박살납니다**
    
    - 딥러닝은 `batch` 단위로 행렬 곱을 돌리면서 GPU를 꽉 채워야 빠름.
        
    - 문장 하나씩 넣으면 연산량이 너무 작아서 GPU가 노는 시간이 많아짐.
        
2. **통계적으로도 배치 학습이 유리**
    
    - Batch Normalization이나 LayerNorm, gradient estimation 등
        
    - 여러 샘플의 손실을 평균내며 미니배치 단위로 학습하는 것이 안정적임.
        

그래서 현실적인 선택은:

> “**여러 문장을 한 텐서에 넣어야 한다 → 길이를 맞춰야 한다**”

그리고 길이를 맞추기 위한 가장 단순·일반적인 해결책:

> **짧은 문장의 뒤를 ‘의미 없는 토큰’으로 채운다 → padding token**

## ✅ 필수 조건 1. Key PAD mask

> 정상 token이 PAD를 보지 못하게 막음

`attention_mask = (input_ids != PAD)  # key 기준`

이건 **항상 필요**.

---

## ✅ 필수 조건 2. Loss mask (이게 제일 중요)

> PAD 위치에서는 **절대 loss를 계산하지 않음**

`loss = (token_loss * (labels != PAD)).sum() / valid_token_count`

이게 없으면 모든 게 무너짐.

---

## ✅ 필수 조건 3. (권장) Embedding layer에서 padding_idx 지정

`nn.Embedding(vocab_size, d, padding_idx=PAD)`

- PAD embedding은 gradient 자체가 0
- “실수 방지용 안전장치”
### Embedding 만으로 안되고, (1번) Attention_Mask도 해야하는 이유

- **이유:** Embedding layer에서 0 벡터로 만들어도, 이후 **Layer Normalization**이나 Linear Layer의 **Bias(편향)** 가 더해지면 **0이 아니게 됩니다.**

- 따라서 **Attention 연산 단계에서 `input_ids != PAD` 마스크로 확실하게 끊어주는 것(Score를 $-\infty$로 보냄)**이 필수입니다.

---


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
