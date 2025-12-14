### 11.3. Least Squares GAN (LSGAN)

- 일반 GAN은 Discriminator에서 **binary cross-entropy(로지스틱)** 손실을 사용.
    
    - 이때 loss 값 범위가 넓고, gradient가 saturation 되기도 쉽다.
        
- LSGAN의 아이디어:
    
    - “진짜=1, 가짜=0” 목표는 그대로 두되,
        
    - loss를 **L2(least squares)** 로 바꾼다.
        마지막 Sigmoid도 삭제.
- 장점:
    
    - 손실 landscape가 더 부드럽고, gradient가 덜 saturate.
        
    - generator가 그라디언트를 더 안정적으로 받게 되어 oscillation(출렁임)이 줄어든다.