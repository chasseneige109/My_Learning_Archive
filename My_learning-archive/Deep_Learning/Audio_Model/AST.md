### Audio Spectrogram Transformer (AST)

"이미지랑 똑같은데, 그냥 ViT 쓰면 안 돼?"라는 아이디어입니다. 
CNN 없이 순수 Attention만 씁니다.

#### (1) Patch Partitioning (패치 자르기)

입력 $X \in \mathbb{R}^{T \times F}$ (예: $1024 \times 128$)를 $P \times P$ 크기의 패치로 자릅니다 (예: $16 \times 16$).

- 패치 개수 ($N$):
    
    $$N = N_t \times N_f = \left(\frac{1024}{16}\right) \times \left(\frac{128}{16}\right) = 64 \times 8 = 512 \text{개}$$
    
- 각 패치는 $16 \times 16 = 256$ 차원의 벡터가 됩니다.
    

#### (2) Linear Projection & Positional Embedding

- **Linear Projection:** $256$차원 벡터를 $d_{model}$(예: 768)로 변환.
    
- **CLS Token 추가:** $N \rightarrow N+1$.
    
- **Positional Embedding:** 오디오는 길이($T$)가 가변적일 수 있습니다. AST는 학습된 2D Positional Embedding을 사용하되, 입력 길이가 바뀌면 **보간(Interpolation)**하여 적용합니다.
    

#### (3) Encoder

이후 과정은 ViT와 완전히 동일합니다.

- Input: $(N+1) \times 768$
    
- Output: `[CLS]` 토큰을 이용해 분류(Classification) 수행.
    

**특징:** CNN을 안 쓰고 전체 스펙트로그램을 **Global**하게 봅니다. 데이터가 많을 때 성능이 뛰어납니다.