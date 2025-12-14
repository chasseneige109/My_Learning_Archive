### 합의(Consensus) 목적과 정지점

- GAN 학습을
    
    - Discriminator 파라미터와 Generator 파라미터 **쌍 (θD,θG)(\theta_D, \theta_G)(θD​,θG​)** 에 대한 **벡터장(vector field)** 로 본다.
        
- Stationary point:
    
    - 두 쪽 gradient 모두 0인 지점.
        
- 아이디어:
    
    - Discriminator와 Generator 둘 다 자신의 gradient norm을 줄이도록 하는 **regularization term** 을 추가.
        
    - 즉, “서로 합의된 상태”에 가도록 유도 → **Consensus objective**.
        
- 하지만:
    
    - gradient가 0인 곳은 **최소, 최대, saddle** 다 포함함.
        
    - 그래서 이 regularization은
        
        - **어떤 정지점(최소 혹은 saddle)** 쪽으로 가게 만드는 경향만 줄 뿐,
            
        - 그게 진짜 ‘의미 있는 최소’인지 보장은 없다.
            
    - 그래도 완전 발산해버리는 것보단 “어딘가에 머물도록” 하는 효과가 있다.