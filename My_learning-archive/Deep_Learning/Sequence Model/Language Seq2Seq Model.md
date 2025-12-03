
## one-hot 기반 embedding 단점

1. vocabulary를 **미리 고정**해야 한다
    
2. 새 단어 추가가 불가능함 (차원을 늘려야 하므로)
    

이건 one-hot의 근본적 한계.

그래서 embedding 자체가 잘 작동해도  
one-hot이라는 기반 representation은 여전히 불편함.

(이건 나중에 subword tokenization으로 해결된다.)