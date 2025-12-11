### In-Context Learning (Zero-shot / Few-shot)

- Zero-shot:
    
    - 단순히 **자연어로 task 설명**만 해주고 정답을 물어봄
        
    - 예: “영어 문장을 프랑스어로 번역해줘” + input만 제시
        
- Few-shot:
    
    - 설명 + **예시 몇 개**를 prompt 안에 넣어 주고
        
    - 마지막에 새로운 input을 줬을 때 output을 생성하게 함
        
- 특징:
    
    - 이때 **모델 파라미터는 업데이트하지 않음** (no gradient)
        
    - 모델이 prompt 안의 예시를 보고 **“on the fly로 규칙을 추론”**하는 듯한 행동
        
    - 이는 거대 Transformer가 scaling 되면서 나타난 대표적인 emergent ability 중 하나로 소개됨