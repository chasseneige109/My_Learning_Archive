
**Cross-Attention (Encoder-Decoder Attention)**은 디코더가 인코더의 정보(Source)를 가져와서 자신의 출력(Target)을 생성하는 핵심 단계입니다.

여기서 가장 중요한 특징은 **행렬의 차원(Dimension)**이 **디코더의 시퀀스 길이($L_{\text{dec}}$)**와 **인코더의 시퀀스 길이($L_{\text{enc}}$)** 사이에서 교차된다는 점입니다.

---

### 0. 사전 정의 (Dimensions)

- $L_{\text{dec}}$: 디코더 현재 시퀀스 길이 (Target Sequence Length)
    
- $L_{\text{enc}}$: 인코더 전체 시퀀스 길이 (Source Sequence Length)
    
- $d_{\text{model}}$: 모델의 임베딩 차원 (예: 512)
    
- $d_k$: 각 헤드의 차원 ($d_{\text{model}} / \text{num\_heads}$, 예: 64)
    

---

### 1. 입력 단계 (Inputs)

두 개의 서로 다른 소스에서 입력이 들어옵니다.

1. **Decoder Input ($\mathbf{X}_{\text{dec}}$):** 디코더의 이전 sublayer(Self-Attention + Add & Norm) 출력
    
    - **행렬 크기:** $(L_{\text{dec}} \times d_{\text{model}})$
        
    - **역할:** **Query ($\mathbf{Q}$)**를 생성하는 재료. (질문자)
        
2. **Encoder Output ($\mathbf{Z}_{\text{enc}}$):** 인코더의 최종 출력입니다.
    
    - **행렬 크기:** $(L_{\text{enc}} \times d_{\text{model}})$
        
    - **역할:** **Key ($\mathbf{K}$)**와 **Value ($\mathbf{V}$)**를 생성하는 재료. (답변자)
        

---

### 2. Q, K ,V 생성

입력 행렬에 학습 가능한 가중치 행렬 $W$를 곱하여 $Q, K, V$를 만듭니다. (Multi-Head 중 하나의 헤드 기준)

- Query ($\mathbf{Q}$): 디코더 입력에서 생성
    
    $$\mathbf{Q} = \mathbf{X}_{\text{dec}} \cdot \mathbf{W}^Q$$
    
    - 연산: $(L_{\text{dec}} \times d_{\text{model}}) \times (d_{\text{model}} \times d_k)$
        
    - **결과 크기:** $\mathbf{Q} \in \mathbb{R}^{L_{\text{dec}} \times d_k}$
        
- Key ($\mathbf{K}$): 인코더 출력에서 생성
    
    $$\mathbf{K} = \mathbf{Z}_{\text{enc}} \cdot \mathbf{W}^K$$
    
    - 연산: $(L_{\text{enc}} \times d_{\text{model}}) \times (d_{\text{model}} \times d_k)$
        
    - **결과 크기:** $\mathbf{K} \in \mathbb{R}^{L_{\text{enc}} \times d_k}$
        
- Value ($\mathbf{V}$): 인코더 출력에서 생성
    
    $$\mathbf{V} = \mathbf{Z}_{\text{enc}} \cdot \mathbf{W}^V$$
    
    - 연산: $(L_{\text{enc}} \times d_{\text{model}}) \times (d_{\text{model}} \times d_k)$
        
    - **결과 크기:** $\mathbf{V} \in \mathbb{R}^{L_{\text{enc}} \times d_k}$
        

---

### 3. 어텐션 스코어 계산 (Scaled Dot-Product)

디코더의 각 토큰(Query)이 인코더의 모든 토큰(Key)과 얼마나 연관이 있는지 내적을 통해 계산합니다.

$$\text{Score} = \frac{\mathbf{Q} \cdot \mathbf{K}^T}{\sqrt{d_k}}$$

- 행렬 연산:
    
    $(L_{\text{dec}} \times d_k) \times (d_k \times L_{\text{enc}}) \rightarrow (L_{\text{dec}} \times L_{\text{enc}})$
    
- **결과 의미:** $(L_{\text{dec}} \times L_{\text{enc}})$ 행렬.
    
    - 행($i$): $i$번째 디코더 토큰이,
        
    - 열($j$): $j$번째 인코더 토큰을 얼마나 주목해야 하는지 나타내는 점수입니다.
        

---
### 3.5 Padding Mask

**Key(인코더 출력)** 쪽의 패딩 위치를 가립니다.
* V쪽은 Key 가중치에서 어차피 0됨.
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \mathbf{M}\right) V$$
### 4. 소프트맥스 (Softmax)

각 디코더 토큰(행)에 대해 확률값(합이 1)으로 변환합니다.

$$\text{Attention Weights} = \text{Softmax}(\text{Score})$$

- **크기 유지:** $(L_{\text{dec}} \times L_{\text{enc}})$
    
- 이 행렬이 바로 **"어텐션 맵(Map)"**입니다. 디코더가 인코더의 어디를 보고 있는지 시각화할 때 이 행렬을 사용합니다.
    

---

### 5. 가중합 (Weighted Sum)

구해진 확률값(가중치)을 실제 정보인 **Value($\mathbf{V}$)**에 곱해서 가져옵니다.

$$\text{Head Output} = \text{Attention Weights} \cdot \mathbf{V}$$

- 행렬 연산:
    
    $(L_{\text{dec}} \times L_{\text{enc}}) \times (L_{\text{enc}} \times d_k) \rightarrow (L_{\text{dec}} \times d_k)$
    
- **결과 의미:**
    
    - 인코더 길이($L_{\text{enc}}$) 차원은 사라지고(Summation), 디코더 길이($L_{\text{dec}}$)만 남습니다.
        
    - 즉, **"인코더의 정보를 요약해서 디코더 각 토큰의 맥락에 맞게 가져온 벡터"**가 됩니다.
        

---

### 6. 최종 출력 (Multi-Head Concatenation & Linear)

여러 헤드($h$개)에서 나온 결과를 합칩니다.

1. **Concat:** $h$개의 $(L_{\text{dec}} \times d_k)$ 행렬을 이어 붙입니다.
    
    - 결과 크기: $(L_{\text{dec}} \times (h \times d_k)) = (L_{\text{dec}} \times d_{\text{model}})$
        
2. Output Linear ($\mathbf{W}^O$): 최종적으로 섞어주는 선형 변환을 수행합니다.
    
    $$\text{Final Output} = \text{ConcatResult} \cdot \mathbf{W}^O$$
    
    - 연산: $(L_{\text{dec}} \times d_{\text{model}}) \times (d_{\text{model}} \times d_{\text{model}}) \rightarrow (L_{\text{dec}} \times d_{\text{model}})$
        

---

### ⚡ 요약: 차원 변화 추적

|**단계**|**입력 행렬 크기**|**출력 행렬 크기**|**비고**|
|---|---|---|---|
|**Input**|Dec: $(L_{\text{dec}}, d_{\text{model}})$<br><br>  <br><br>Enc: $(L_{\text{enc}}, d_{\text{model}})$|-|두 입력의 길이가 다를 수 있음|
|**Q, K, V 생성**|-|$Q: (L_{\text{dec}}, d_k)$<br><br>  <br><br>$K, V: (L_{\text{enc}}, d_k)$|Q는 디코더, K/V는 인코더 출신|
|**$QK^T$ (Score)**|$Q, K$|$(L_{\text{dec}}, L_{\text{enc}})$|직사각형 행렬 (Attention Map)|
|**Attention Output**|Weights, $V$|$(L_{\text{dec}}, d_k)$|인코더 길이는 사라짐|
|**Final Output**|All Heads|**$(L_{\text{dec}}, d_{\text{model}})$**|디코더 입력과 동일한 형태 복구|

결국 Cross-Attention은 **디코더의 길이($L_{\text{dec}}$)를 유지**하면서, 내용은 **인코더($L_{\text{enc}}$)의 정보를 함축**하여 채워 넣는 과정입니다.