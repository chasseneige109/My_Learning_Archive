
## one-hot 기반 embedding 단점

1. vocabulary를 **미리 고정**해야 한다
    
2. 새 단어 추가가 불가능함 (차원을 늘려야 하므로)
    

이건 one-hot의 근본적 한계.

그래서 embedding 자체가 잘 작동해도  
one-hot이라는 기반 representation은 여전히 불편함.

(이건 나중에 subword tokenization으로 해결된다.)


네, **언어 모델(Language Model)**, 특히 **LSTM 기반의 다음 단어 예측(Next Word Prediction) 모델**이 어떻게 학습되고 추론하는지, **행렬 연산(Matrix Operation) 수준**에서 A부터 Z까지 해부해 드리겠습니다.

우리의 목표는 **"I love AI"**라는 문장을 학습하고, 다시 생성해내는 것입니다.

---

### 0. 셋팅 (The Setup) 🛠️

먼저 행렬들의 크기(Dimension)를 정의합시다.

- **$V$ (Vocabulary Size):** 10,000개 (단어장 크기)
    
- **$D$ (Embedding Size):** 128차원 (단어 벡터 크기)
    
- **$H$ (Hidden Size):** 256차원 (LSTM 내부 메모리 크기)
    

#### 1) 준비된 행렬들 (Parameters)

1. **임베딩 행렬 $\mathbf{E}$:** $(V \times D) = (10000 \times 128)$
    
2. **LSTM 가중치 $\mathbf{W}_{lstm}$:** 입력과 은닉 상태를 처리하는 4개의 게이트용 가중치 통합본.
    
    - 입력 처리용: $(D \times 4H) = (128 \times 1024)$
        
    - 은닉 처리용: $(H \times 4H) = (256 \times 1024)$
        
    - 편향 $\mathbf{b}_{lstm}$: $(1 \times 4H) = (1 \times 1024)$
        
3. **출력 투영 행렬 $\mathbf{W}_{proj}$:** $(H \times V) = (256 \times 10000)$
    
4. **출력 편향 $\mathbf{b}_{proj}$:** $(1 \times V) = (1 \times 10000)$