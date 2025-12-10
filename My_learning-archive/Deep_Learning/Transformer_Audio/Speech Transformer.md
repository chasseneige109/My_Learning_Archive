### Speech Transformer (Convolution Subsampling)

초기 음성 인식 모델입니다. $T$가 너무 길면(예: 3000 프레임) Self-Attention의 연산량($O(T^2)$)이 감당이 안 됩니다. 그래서 **Convolution으로 길이를 강제로 줄이고 시작**합니다.

#### (1) Convolution Subsampling (입력 압축)

입력 스펙트로그램 $X \in \mathbb{R}^{B \times 1 \times T \times F}$ (Batch, Channel, Time, Freq)가 들어옵니다.

1. **Conv2D Layer 1:**
    
    - 커널: $3 \times 3$, **Stride: $(2, 2)$** (시간과 주파수를 1/2로 줄임)
        
    - 출력: $B \times C_{out} \times (T/2) \times (F/2)$
        
2. **Conv2D Layer 2:**
    
    - 커널: $3 \times 3$, **Stride: $(2, 2)$** (또 1/2로 줄임)
        
    - 출력: $B \times C_{out} \times (T/4) \times (F/4)$
        

#### (2) Reshape & Projection (Transformer 입력화)

이제 4차원 텐서를 Transformer가 먹을 수 있는 3차원 시퀀스로 폅니다.

- **Flatten:** 주파수 축($F/4$)과 채널($C_{out}$)을 합칩니다.
    
    - $X_{flat} \in \mathbb{R}^{B \times (T/4) \times (C_{out} \cdot F/4)}$
        
- **Linear Projection:** 모델의 Hidden Dimension($d_{model}$)으로 차원을 맞춥니다.
    
    - $X_{enc} = X_{flat} W + b \quad (W \in \mathbb{R}^{(C_{out} \cdot F/4) \times d_{model}})$
        
    - 최종 입력: **$B \times (T/4) \times d_{model}$**
        

**핵심:** 시간 축 $T$를 $T/4$로 줄여 연산량을 확보하고, 지역적 특징(Local Feature)을 Convolution으로 미리 추출했습니다.