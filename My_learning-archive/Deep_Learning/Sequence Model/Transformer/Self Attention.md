### 각 단어가 모든 단어를 직접 “쳐다보게” 하자

각 단어에 대해:

- hidden representation h를 하나 만든 뒤
    
- 거기서 세 개를 뽑음:
    
    qi=WQhi,ki=WKhi,vi=WVhiq_i = W_Q h_i,\quad k_i = W_K h_i,\quad v_i = W_V h_iqi​=WQ​hi​,ki​=WK​hi​,vi​=WV​hi​

그다음, 단어 i의 새 표현을 만들 때:

1. i의 query qiq_iqi​와 모든 단어의 key kjk_jkj​를 내적해서 **점수(score)** 계산
    
    sij=qi⊤kjs_{ij} = q_i^\top k_jsij​=qi⊤​kj​
2. softmax를 씌워서 **attention weight**로 만듦
    
    aij=softmaxj(sij)a_{ij} = \text{softmax}_j(s_{ij})aij​=softmaxj​(sij​)
3. 그 weight로 value들을 가중합
    
    hi′=∑jaijvjh'_i = \sum_j a_{ij} v_jhi′​=j∑​aij​vj​

→ 이렇게 하면 **각 단어의 새 벡터 hi′h'_ihi′​**는  
“어떤 단어를 얼마나 봤는지”에 따라 달라진다 = **문맥 반영**.

이게 **single-head self-attention**.