### 1. 핵심 아이디어: "Hard Prompt"에서 "Soft Prompt"로의 진화

Prompt Tuning을 이해하려면 먼저 **Prompt**의 개념 변화를 알아야 합니다.

- **Hard Prompt (Discrete):** 사람이 직접 텍스트로 작성한 명령어입니다.
    
    - 예: "Translate this to English: [Input Sentence]"
        
    - **문제점:** "Translate"라고 쓸지, "To English"라고 쓸지 단어 하나만 바꿔도 성능이 널뛰기합니다. (최적화가 불가능한 **이산 공간(Discrete Space)**의 문제)
        
- **Soft Prompt (Continuous):** Prompt Tuning이 사용하는 방식입니다.
    
    - 사람의 언어(단어)가 아니라, **실수(Float)로 이루어진 벡터**를 사용합니다.
        
    - 역전파(Backpropagation)를 통해 **연속 공간(Continuous Space)**에서 최적의 벡터 값을 찾습니다.
        
    - **의미:** 인간의 사전에는 없는 단어지만, 모델에게는 **"이 작업을 수행해!"라고 완벽하게 이해되는 추상적인 신호**를 찾아내는 과정입니다.
        

---

### 2. 구조적 차이: 입력 레벨 주입 (Input-Level Injection)

Prefix Tuning과 가장 큰 차이점은 **간섭의 깊이**입니다.

#### (1) Prefix Tuning (Deep Modification)

- **위치:** 모든 Transformer Block의 Attention $K, V$ 앞.
    
- **비유:** 요리사(모델)에게 레시피를 줄 때, 재료 손질 단계, 굽는 단계, 플레이팅 단계 **마다 옆에서 계속 "이렇게 해"라고 지시**하는 것.
    
- **특징:** 모델 통제력이 강력하지만, 파라미터가 상대적으로 많고 구현이 까다롭습니다.
    

#### (2) Prompt Tuning (Shallow Modification)

- **위치:** 오직 **Embedding Layer(입력층)** 앞.
    
- 행렬 연산:
    
    입력 문장 $X$가 임베딩된 행렬을 $E \in \mathbb{R}^{T \times d}$라고 하고, 학습 가능한 프롬프트 행렬을 $P \in \mathbb{R}^{L \times d}$라고 할 때, 모델에 들어가는 최종 입력은 단순한 Concatenation입니다.
    
    $$\text{Input} = [P; E] \in \mathbb{R}^{(L+T) \times d}$$
    
- **비유:** 요리사에게 요리 시작 전에 **"이건 VIP 손님용 한식 코스야"라고 딱 한 번 말해주고**, 주방(모델 내부)에는 일절 들어가지 않는 것.
    
- **특징:**
    
    - 모델 내부 구조를 전혀 건드리지 않습니다.
        
    - 역전파가 모델의 맨 끝단에서 입력층($P$)까지 아주 멀리 흘러와야 하므로 학습 난이도가 조금 더 있습니다.
        

---

### 3. 왜 작동할까? (The Power of Scale)

여기서 중요한 의문이 생깁니다. **"입력에서만 살짝 건드려주는데, 깊은 레이어를 통과하면서 그 정보가 희석되지 않을까?"**

Prompt Tuning 논문의 핵심 발견은 바로 **"모델이 거대할수록(Scale) 잘 작동한다"**는 것입니다.

- **Small Model (예: BERT, GPT-2 small):** 입력에서만 힌트를 주면, 레이어를 지나면서 정보가 잊혀지거나 왜곡되어 성능이 Full Fine-tuning보다 현저히 떨어집니다.
    
- **Large Model (예: 10B+ Parameters):** 모델이 거대해지면, 입력단에서의 작은 섭동(Perturbation)이나 가이드만으로도 **원하는 목적지까지 문맥을 유지하는 능력**이 생깁니다.
    
- **결과:** 모델 사이즈가 커질수록 Prompt Tuning의 성능이 Full Fine-tuning 성능에 수렴합니다.
    

### 4. Initialization (초기화)의 중요성

Prompt 벡터 $P$는 랜덤 초기화(Random Initialization)를 하면 학습이 잘 안되거나 매우 오래 걸립니다. (최적화 관점에서 Local Minima에 빠지기 쉽습니다.)

그래서 보통 **의미 있는 값**으로 초기화합니다.

1. **Class Label:** "요약", "번역", "감정분석" 같은 단어의 임베딩 값을 가져와서 초기값으로 씁니다.
    
2. **Sample Text:** 데이터셋에서 자주 등장하는 단어들의 임베딩을 초기값으로 씁니다.
    

이렇게 하면 최적화의 **시작점(Starting Point)**이 정답 근처에 위치하게 되어 수렴 속도와 성능이 좋아집니다.

### 요약

**Prompt Tuning은...**

1. **Prefix Tuning의 경량화 버전:** 모든 레이어가 아닌 **입력층**에만 가상 토큰을 붙입니다.
    
2. **Continuous Optimization:** 인간이 프롬프트를 고민하는 대신, **역전파가 최적의 프롬프트 벡터를 찾게** 시킵니다.
    
3. **Scale-Dependent:** 모델이 작을 땐 성능이 별로지만, **모델이 클수록(LLM)** Full Fine-tuning에 버금가는 성능을 내면서 메모리는 1/1000도 안 쓰게 됩니다.
    

작성자님이 공부하시는 **거대 모델 트렌드**에서는 Prefix Tuning보다 구현이 쉽고 효과적인 Prompt Tuning(또는 그 변형들)이 더 자주 언급되는 추세입니다.