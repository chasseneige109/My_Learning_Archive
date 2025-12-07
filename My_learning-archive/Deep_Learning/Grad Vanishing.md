# Gradient Vanishing의 보안책

현대 딥러닝 아키텍처는  
“gradient vanishing이 생길 수 있는 모든 지점마다  
각각의 ‘국소적(local) 해결책’을 체계적으로 배치한 구조

## 1. Residual Connection

- Self-attention, FFN은 모두 **비선형 + 선형 조합**
    
- Jacobian의 spectral norm < 1 → gradient 감소


## 2. LayerNorm: “분산 붕괴/폭발 방지기”

## 3. Self-Attention 자체의 안정화 설계

QK^T 내적하고 root(d_K)로 나누기