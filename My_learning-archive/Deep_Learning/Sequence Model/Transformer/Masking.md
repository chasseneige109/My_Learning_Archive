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

✅ 이게 없으면 모든 게 무너짐.

---

## ✅ 필수 조건 3. (권장) Embedding layer에서 padding_idx 지정

`nn.Embedding(vocab_size, d, padding_idx=PAD)`

- PAD embedding은 gradient 자체가 0
    
- “실수 방지용 안전장치”
    

---

## ✅ 권장 조건 4. Encoder/Decoder 모두에서 mask 일관성 유지

- Encoder self-attention
    
- Decoder self-attention
    
- Cross-attention
    

👉 **모든 attention block에서 PAD mask 적용**



# Casual Mask

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

QK^T 를 내적하고 softmax에 넣기