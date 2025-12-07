
## ✅ Gumbel 분포

### 정의

표준 Gumbel(0, 1) 분포:

G=−log⁡(−log⁡(U)), U∼Uniform(0,1)

이 분포는 중요한 성질을 가짐.

---

## ✅ Gumbel-Max Trick (이산 샘플링)

다음은 **정확한 categorical sampling**입니다:

y = arg⁡max⁡i (log⁡pi + gi) , gi ∼ Gumbel (0,1)

✅ 매우매우 놀랍게도:

Pr⁡(y=i) = pi

즉,

> **Gumbel noise + argmax = categorical sampling** 이 기가막히게도..! 성립함.

- Argmax를 softmax + log p + Gumbel noise 로 근사함.

$$y_i = \frac{\exp((\log(\pi_i) + g_i) / \tau)}{\sum_{j=1}^{K} \exp((\log(\pi_j) + g_j) / \tau)}$$
- $\pi_i$: 기존 Logit의 확률값
    
- $g_i$: Gumbel Noise (샘플링의 무작위성을 담당)
    
- **$\tau$ (Temperature): 핵심 파라미터**
    

### $\tau$ (온도)의 역할

- **$\tau \rightarrow \infty$:** 출력이 Uniform Distribution(모든 확률이 비슷)에 가까워짐.
    
- **$\tau \rightarrow 0$:** 출력이 **One-hot Vector (Argmax)** 에 매우 가까워짐.
    

핵심 트릭:

학습 초기에는 $\tau$를 높게 잡아서 부드럽게(Soft) 학습하다가, 
학습이 진행될수록 $\tau$를 0에 가깝게 줄여서 실제 Argmax(Discrete)와 거의 똑같이 동작하게 함. 
이렇게 하면 **미분 가능성(Differentiability)**을 유지하면서도 
이산적인 선택(Discrete Choice)**을 흉내 낼 수 있습니다.


## ✅ Gumbel trick이 들어가는 위치 / 순서

### 1. 일반 Language Model 학습 (Softmax + CE)

`h → logits z → softmax → p(word)`

`p_i = exp(z_i) / Σ_j exp(z_j)`

- softmax 씌운 걸, 정답과 함께 Loss에 넣음.
### 2. Gumbel-Softmax를 쓰는 경우 (샘플이 필요할 때)

> 🎯 **목적: 단어를 “뽑으면서도” gradient를 흘리고 싶다**

`h → logits z   → z + gumbel_noise   → (z + g) / τ   → softmax   → y (soft sample)`
`y_i = softmax( (z_i + g_i) / τ )`

 이 softmax 출력이 바로 **Gumbel-Softmax 샘플**
- Gumbel softmax 를 정답과 함께 Loss에 넣음.


## ✅ 언제 쓰는가? (Use Cases)

“Gumbel-Softmax는 softmax보다 더 one-hot에 가까운 출력을 만들어  
이산 선택을 연속적으로 근사한다.  
이 출력은 downstream loss에서 정답과 비교될 수 있지만,  
언어모델처럼 분포 자체를 학습하는 문제에는 필요하지 않다.

### 🎯모델 내부🎯에서 ‘무조건 하나를 골라야 하는’ 명확한 이산 결정 문제가 있을 때 사용된다. 
### 모델 밖의 '출력'과는 다름. 

1. **Text GANs (Generative Adversarial Networks):**
    
    - Generator가 문장을 만들고 Discriminator가 검사하는 구조에서, 두 네트워크 사이를 미분 가능하게 연결하고 싶을 때. (원래 텍스트는 불연속적이라 GAN 적용이 어렵거든요.)
        
2. **Discrete VAE (Variational Autoencoder):**
    
    - Latent Variable(잠재 변수) $z$를 연속적인 가우시안 분포가 아니라, **카테고리(Discrete) 분포**로 쓰고 싶을 때. (예: DALL-E 1의 dVAE)
        
3. **Hard Attention Mechanism:**
    
    - Soft Attention(모든 걸 조금씩 봄)이 아니라, **"딱 하나만 본다"**는 메커니즘을 학습시키고 싶을 때.
        
4. **Neural Architecture Search (NAS):**
    
    - 네트워크의 연결 구조를 "연결한다/안 한다(0 or 1)"로 선택하는 과정을 미분 가능하게 만들어 경사 하강법으로 구조를 찾을 때.
