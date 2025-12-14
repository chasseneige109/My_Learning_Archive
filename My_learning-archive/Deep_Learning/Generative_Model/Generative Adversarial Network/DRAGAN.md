### 11.4. DRAGAN (CMU에서 나온 방법)

- 목표:
    
    - Real 데이터 근처에서 Discriminator의 **gradient norm을 작게** 만들어  
        함수가 **매우 smooth** 하도록.
        
- 이유:
    
    - Discriminator가 각 real 데이터 포인트 주변에만 sharp peak를 만들고,
        
    - 나머지 공간은 아무 정보가 없으면,
        
    - generator가 그 peak 근처에 도달하기 전까지는 gradient를 거의 못 받는다.
        
- 방법:
    
    - Real 데이터 근처의 주변 점들을 샘플링하여
        
    - 해당 위치들에서 Discriminator gradient norm이 커지지 못하게 regularization.
        
- 효과:
    
    - Real 데이터 주변이 **완만한 지형**이 되어
        
    - generator가 그쪽으로 이동하면서 점점 의미 있는 신호를 받게 됨.