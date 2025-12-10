**Vision Transformer (ViT)**는 2021년 구글 리서치 팀("An Image is Worth 16x16 Words")이 발표한 논문으로, **"CNN 없이도 이미지를 처리할 수 있다"**는 것을 증명하여 컴퓨터 비전(Computer Vision)의 패러다임을 완전히 바꾼 모델입니다.

사용자님께서 익숙하신 **행렬/텐서 변환 관점**에서 ViT의 파이프라인을 깊이 있게 해부해 드리겠습니다.

---

### 1. 핵심 철학: "이미지도 문장이다"

NLP의 Transformer는 문장을 **토큰(단어)의 시퀀스**로 봅니다. ViT는 이미지를 **패치(Patch)의 시퀀스**로 봅니다.

- **NLP:** "I love AI" $\rightarrow$ `["I", "love", "AI"]` (토큰 3개)
    
- **ViT:** 고양이 사진 $\rightarrow$ `[패치1, 패치2, ..., 패치9]` (이미지 조각들)
    

---

### 2. ViT 파이프라인 상세 분석 (Matrix Transformation)

이미지 $x \in \mathbb{R}^{H \times W \times C}$가 들어왔을 때, Transformer Encoder에 들어가기까지의 과정을 단계별로 쪼개보겠습니다. (예: $224 \times 224$ RGB 이미지, 패치 크기 $16 \times 16$)

#### Step 1: Patch Partitioning (이미지 조각내기)

이미지를 $P \times P$ 크기의 패치로 자릅니다.

- **패치 개수 ($N$):** $N = (H \times W) / (P \times P) = (224^2 / 16^2) = 196$개
    
- **결과:** $N$개의 이미지 조각들.
    

#### Step 2: Linear Projection of Flattened Patches (임베딩)

각 패치를 **벡터**로 만듭니다.

1. **Flatten:** 각 패치($16 \times 16 \times 3$)를 1차원 벡터로 폅니다.
    
    - 벡터 크기: $16 \times 16 \times 3 = 768$
        
    - 입력 형태: $N \times 768$ (196개의 768차원 벡터)
        
2. **Linear Projection ($E$):** 이 벡터를 Transformer의 Hidden Dimension ($D$, 예: 768)으로 매핑하는 행렬 $E$를 곱합니다.
    
    - $x_p = x_{flatten} \times E$
        
    - 이것이 **Patch Embedding**입니다. (NLP의 Word Embedding과 동일한 역할)
        

#### Step 3: `[CLS]` 토큰과 Positional Embedding 추가

1. **Learnable `[CLS]` Token:**
    
    - BERT처럼, **분류(Classification)를 위한 전용 토큰** 하나를 맨 앞에 붙입니다.
        
    - 시퀀스 길이 변화: $N \rightarrow N+1$ (196 + 1 = 197개)
        
2. **Positional Embedding:**
    
    - Transformer는 순서 개념이 없으므로(Permutation Invariant), 위치 정보를 더해줍니다.
        
    - **중요:** 2D 이미지지만, 보통 **학습 가능한 1D 포지션 임베딩(Learnable 1D Position Embedding)**을 사용합니다.
        
    - 모델이 학습하면서 "아, 이 포지션 임베딩을 가진 패치는 저거랑 위아래 관계구나"를 스스로 깨닫습니다.
        
    - 최종 입력: $(N+1) \times D$
        

#### Step 4: Transformer Encoder

이제 준비된 $(N+1) \times D$ 행렬을 **Standard Transformer Encoder**에 넣습니다.

- **MSA (Multi-Head Self-Attention):** 모든 패치가 다른 모든 패치를 봅니다. (**Global Receptive Field**)
    
- **MLP (Multi-Layer Perceptron):** 각 패치별로 특징을 추출합니다.
    
- **LN (Layer Norm) & Residual:** 학습 안정화.
    

#### Step 5: Classification Head

마지막 층에서 나온 $N+1$개의 벡터 중, 맨 앞의 **`[CLS]` 토큰에 해당하는 벡터** 하나만 뽑아서 MLP에 넣고 최종 클래스를 예측합니다.

---

### 3. CNN vs ViT: Inductive Bias의 전쟁

이 부분이 ViT를 이해하는 데 가장 중요한 이론적 배경입니다.

| **구분**              | **CNN (ResNet)**                                                                                                  | **ViT (Vision Transformer)**   |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| **Inductive Bias**  | **강함 (High)**                                                                                                     | **약함 (Low)**                   |
| **가정 (Assumption)** | 1. **Locality:** 근처 픽셀끼리 관계있다.<br><br>  <br><br>2. **Translation Invariance:** 고양이가 왼쪽 위에 있든 오른쪽 아래에 있든 똑같은 고양이다. | 가정 없음. (모든 픽셀 관계를 처음부터 배워야 함)  |
| **데이터 효율성**         | 적은 데이터로도 학습 잘 됨 (규칙이 내장됨)                                                                                         | **엄청난 데이터(JFT-300M 등)가 필요함**   |
| **Receptive Field** | 초기 레이어는 좁음 $\rightarrow$ 깊어질수록 넓어짐                                                                                | **첫 레이어부터 이미지 전체를 봄 (Global)** |

- **왜 작은 데이터셋에서는 CNN이 이기는가?**
    
    - CNN은 정답을 맞히기 위한 '치트키(Locality)'를 가지고 시작합니다. ViT는 맨땅에 헤딩하며 "아, 바로 옆 픽셀이 중요한 거구나"라는 것부터 배워야 하므로, 데이터가 적으면 과적합(Overfitting)되거나 학습이 안 됩니다.
        
- **왜 대형 데이터셋에서는 ViT가 압도하는가?**
    
    - CNN의 강한 가정(Bias)은 데이터가 많아지면 오히려 **제약(Limit)**이 됩니다. ViT는 제약이 없어서, 데이터가 충분하면 CNN이 보지 못하는 **먼 거리의 픽셀 관계(Long-range Dependency)**까지 학습하여 더 높은 성능(Capacity)을 냅니다.
        

---

### 4. 내부 해석: Mean Attention Distance

논문에서 분석한 내용 중 흥미로운 것은 레이어 깊이에 따른 **Attention Distance(얼마나 멀리 있는 패치를 보는가)**입니다.

- **초기 레이어 (Low Layers):**
    
    - 어떤 헤드는 가까운 곳(Local)을 보고, 어떤 헤드는 먼 곳(Global)을 봅니다.
        
    - CNN은 초기 레이어에서 무조건 가까운 것만 볼 수 있는 반면, **ViT는 처음부터 전체를 볼 수 있는 능력**이 있습니다.
        
- **후기 레이어 (Deep Layers):**
    
    - 대부분의 헤드가 이미지 전체(Global)를 넓게 보며 의미론적(Semantic) 정보를 통합합니다.
        

---

### 5. 로보틱스(Physical AI) 관점에서의 시사점

사용자님의 전공인 **로보틱스**에서 ViT는 현재 **사실상의 표준(De Facto Standard)**이 되어가고 있습니다. (예: RT-2, PaLM-E 같은 VLA 모델)

1. **Multi-modal 통합 용이성:**
    
    - 텍스트도 토큰, 이미지도 토큰(패치)으로 처리하므로, **LLM과 결합**하기가 CNN보다 훨씬 구조적으로 자연스럽습니다.
        
2. **전역 문맥 파악 (Global Context):**
    
    - 로봇이 방 안의 상황을 판단할 때, CNN처럼 국소적인 특징(모서리, 질감)을 단계적으로 합치는 것보다, ViT처럼 **한 번에 방 전체의 관계(책상 위의 컵과 저 멀리 있는 사람의 관계)**를 파악하는 것이 유리할 때가 많습니다.
        

**요약하자면:** ViT는 **"이미지의 지역적 특성이라는 고정관념(Inductive Bias)을 버리고, 압도적인 데이터와 연산량으로 픽셀 간의 모든 관계를 학습해 버리는 모델"**입니다.