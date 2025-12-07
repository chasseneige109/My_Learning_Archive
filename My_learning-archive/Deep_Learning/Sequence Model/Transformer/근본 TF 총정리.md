### Phase 1. 입력 처리 (Input Processing)

> 모델에 들어가기 전, 숫자로 변환하고 위치 정보를 입히는 단계

#### 1. Tokenization & Input Embedding

- 자연어 문장을 토큰으로 자르고, 각 토큰을 $d_{model}$ 차원의 벡터(Dense Vector)로 변환합니다.
    
- **Scaling:** 이 벡터들에 $\sqrt{d_{model}}$을 곱해줍니다. (이후 더해질 PE값에 정보가 묻히지 않게 크기를 키워주는 테크닉)
    

#### 2. Positional Encoding (PE)

- **문제:** 트랜스포머는 병렬 처리(Parallel)를 하므로, "I love AI"와 "AI love I"를 구분하지 못합니다. (순서 정보 부재)
    
- **해결:** 사인(Sine), 코사인(Cosine) 함수를 이용해 각 위치(pos)마다 고유한 패턴을 가진 벡터를 만듭니다.
    
- **연산:** `Input Embedding + Positional Encoding` (Concatenate가 아니라 **Element-wise Sum**입니다).
    

---

### Phase 2. 인코더 (Encoder Block) x N

> 문맥을 이해하고 압축하는 단계. 이 블록이 보통 6개(N=6) 쌓여 있습니다.

#### 3. Multi-Head Self-Attention (Encoder)

- **입력:** Phase 1의 결과물.
    
- **동작:** 입력 문장 내의 단어들끼리 서로의 관계(Attention Score)를 계산합니다.
    
- **Masking:** **Padding Mask** 적용 (문장의 빈 공간인 `<PAD>` 토큰은 점수를 0으로 만들어 무시).
    
- **결과:** 문맥 정보가 반영된 벡터 시퀀스.
    

#### 4. Add (Residual Connection)

- **동작:** `Attention 결과 + 원래 입력(3번 들어가기 전)`
    
- **이유:** 기울기 소실(Vanishing Gradient) 방지 및 정보 보존. "변화된 정보만 학습해라"라는 의미.
    

#### 5. Layer Normalization

- **동작:** 각 샘플(토큰) 별로 평균과 분산을 구해 정규화합니다. ($\frac{x - \mu}{\sigma}$)
    
- **특징:** 배치 크기(Batch size)와 무관하게 동작하며 학습을 안정화합니다.
    

#### 6. Position-wise Feed-Forward Network (FFN)

- **구조:** `Linear(확장) -> ReLU(활성화) -> Linear(축소)`
    
- **동작:** 보통 $d_{model}$(512) 차원을 $4 \times d_{model}$(2048)로 뻥튀기했다가 다시 줄입니다.
    
- **이유:** Attention이 "관계"를 본다면, FFN은 각 토큰이 가진 "정보 자체"를 가공하고 정리하는 역할을 합니다. **비선형성(Non-linearity)**을 추가해 모델의 표현력을 높입니다.
    

#### 7. Add & LayerNorm

- FFN의 결과에 다시 **Residual Connection(Add)**을 하고 **LayerNorm**을 수행합니다.
    
- **★ Encoder 최종 출력:** 이 결과값($K, V$)은 Decoder로 전달됩니다.
    

---

### Phase 3. 디코더 (Decoder Block) x N

> 번역(생성)을 수행하는 단계. 학습 시(Teacher Forcing)와 추론 시 동작이 다릅니다.

#### 8. Output Embedding & PE

- **입력:** (학습 시) 정답 문장 전체(Shifted Right), (추론 시) 현재까지 생성된 문장.
    
- 역시 Embedding 후 **Positional Encoding**을 더해줍니다.
    

#### 9. Masked Multi-Head Self-Attention

- **동작:** Decoder 입력 안에서 자기들끼리 관계를 봅니다.
    
- **Masking:**
    
    1. **Padding Mask:** 배치 내 짧은 문장의 뒤쪽 무시.
        
    2. **Causal Mask:** 자기보다 미래에 있는 단어 무시 (삼각형 마스크).
        
- 이 두 마스크의 교집합을 적용합니다.
    

#### 10. Add & LayerNorm

- Self-Attention 결과에 `Add(잔차 연결)` 및 `Norm` 수행.
    
- **이 결과가 Cross Attention의 Query($Q$)가 됩니다.**
    

#### 11. Multi-Head Cross Attention

- **재료:**
    
    - **Query ($Q$):** 방금 10번을 통과한 Decoder의 상태.
        
    - **Key ($K$), Value ($V$):** 아까 Phase 2(7번)에서 끝난 **Encoder의 최종 출력**.
        
- **동작:** $Q$를 이용해 $K$를 탐색하고, $V$를 가중합(Weighted Sum)하여 가져옵니다.
    
- **Masking:** **Encoder Padding Mask** 적용 (Encoder 쪽 문장의 `<PAD>` 부분을 쳐다보지 않음).
    

#### 12. Add & LayerNorm

- Cross Attention 결과에 `Add` 및 `Norm`.
    

#### 13. Position-wise Feed-Forward Network (FFN)

- Encoder와 동일 구조. 가져온 정보를 최종적으로 정리합니다.
    

#### 14. Add & LayerNorm

- FFN 결과에 `Add` 및 `Norm`.
    
- 이것이 Decoder 블록 하나의 출력이며, 다음 Decoder 블록으로 넘어갑니다. (N번 반복)
    

---

### Phase 4. 최종 출력 (Final Output)

> 사람이 이해할 수 있는 확률값으로 변환

#### 15. Linear Projection

- Decoder의 최종 출력($d_{model}$ 사이즈)을 **단어 집합의 크기(Vocab Size)**만큼 넓힙니다. (예: 512 -> 30,000)
    
- 각 단어(Logits)가 될 점수를 계산하는 과정입니다.
    

#### 16. Softmax

- Linear의 출력값(Logits)을 0~1 사이의 확률값으로 변환합니다. 합은 1이 됩니다.
    
- 여기서 가장 확률이 높은 단어를 선택(Argmax)하거나 샘플링하면 최종 단어가 나옵니다.
    

---

### ✅ 요약: 놓치기 쉬운 포인트 체크

1. **Add(Residual)**는 모든 Sub-layer(Attention, FFN) 뒤에 반드시 붙습니다.
    
2. **LayerNorm**도 모든 Add 뒤에 반드시 붙습니다. (Post-LN 기준)
    
3. **Cross Attention**의 $K, V$는 Encoder에서 오고, $Q$는 Decoder에서 옵니다.
    
4. **FFN**은 차원을 키웠다가(x4) 줄이는 과정입니다.
    
5. **PE(위치 인코딩)**은 입력 벡터에 "더해지는" 것이지 붙이는(Concat) 게 아닙니다.
    

이 순서가 머릿속에 파이프라인처럼 연결되어 흐르면 Transformer 구조를 완벽하게 장악하신 겁니다.