### Conformer (Convolution-augmented Transformer)

현재 음성 인식(ASR) 분야의 **De Facto Standard(표준)**입니다.

Transformer는 전역적(Global)인 건 잘 보는데, 미세한 발음 차이 같은 국소적(Local) 패턴을 놓칩니다. CNN은 반대죠. 그래서 둘을 샌드위치처럼 섞었습니다.

구조는 **Macaron Net** 스타일을 따릅니다. (FFN을 반으로 쪼개서 앞뒤에 배치)

$$x_{out} = \text{FFN}(\text{Conv}(\text{MHSA}(\text{FFN}(x))))$$

여기서 가장 중요한 **Convolution Module**의 내부 행렬 연산을 뜯어보겠습니다.

#### Conformer Block 내부의 Convolution Module 상세

입력 $x \in \mathbb{R}^{T \times d_{model}}$이 들어옵니다.

1. **Pointwise Conv (Expansion):**
    
    - $1 \times 1$ Conv를 사용하여 채널(차원)을 2배로 뻥튀기합니다.
        
    - $x \in \mathbb{R}^{T \times d} \rightarrow \mathbb{R}^{T \times 2d}$
        
    - 이유: **GLU(Gated Linear Unit)** 활성화 함수를 쓰기 위해서입니다.
        
2. **GLU (Gated Linear Unit):**
    
    - 채널을 절반($A, B$)으로 나눕니다.
        
    - $\text{Output} = A \otimes \sigma(B)$ ($\sigma$는 시그모이드)
        
    - 정보($A$)를 얼마나 통과시킬지($B$)를 스스로 결정하는 게이트 역할을 합니다.
        
    - 다시 $\mathbb{R}^{T \times d}$로 돌아옵니다.
        
3. **Depthwise Conv (Local Pattern):**
    
    - **여기가 핵심입니다.** 채널별로 독립적인 1D Conv를 수행합니다.
        
    - 커널 크기 $K$ (예: 31). 자신의 앞뒤 $K$개의 프레임만 봅니다.
        
    - **역할:** "지금 이 프레임이 앞뒤 문맥이랑 부드럽게 이어지는가?" (Local Dependency)
        
4. **Batch Norm & Swish:** 정규화 및 활성화.
    
5. **Pointwise Conv (Projection):**
    
    - $1 \times 1$ Conv로 최종 차원을 정리하고 섞어줍니다.
        
    - $\mathbb{R}^{T \times d} \rightarrow \mathbb{R}^{T \times d}$