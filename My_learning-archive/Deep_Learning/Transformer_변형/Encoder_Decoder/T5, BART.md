# T5 - **Text-to-Text Transfer Transformer 

- BERT(Encoder)와 GPT(Decoder)를 결합

- "Encoder가 소화시킨 정보를 Decoder가 어떻게 가져다 쓰는가?" 
--> Cross Attention


# A. T5의 경우 (Span Corruption)

T5는 모든 문제를 **"텍스트 $\rightarrow$ 텍스트"**로 변환하는 단일 프레임워크를 사용합니다.

- **학습 목표:** 입력 텍스트의 여러 부분을 `[MASK]` 대신 **`[SENTINEL]`** 토큰으로 가리고, Encoder가 이를 이해한 후 Decoder가 가려진 스팬(Span)들을 복구하도록 합니다.
    
    - **입력:** `The man [SENTINEL 1] went to [SENTINEL 2].`
        
    - **정답:** `[SENTINEL 1] bought a sandwich [SENTINEL 2] the store.`
        
- **NSP 대체 효과:** 문장의 연속된 부분을 복구하는 과정에서 문맥적 이해와 흐름 파악 능력을 동시에 학습하게 되므로, NSP의 목표가 자연스럽게 포함됩니다.

# B. BART의 경우 (Denoising)

BART는 BERT의 Encoder와 GPT의 Decoder를 결합한 형태로, Encoder가 다양한 형태로 훼손된 문장을 보고 Decoder가 원문을 **재구성**하도록 학습합니다.

- **다양한 훼손:** 문장을 임의로 섞거나(Sentence Shuffling), 문장 전체를 삭제(Sentence Deletion)하거나, 마스킹(Masking)을 합니다.
    
- **NSP 대체 효과:** 특히 **문장 순서 섞기(Sentence Shuffling)** 태스크는 모델에게 문장 간의 논리적 흐름과 배열을 이해하도록 강제하기 때문에, NSP가 목표했던 **장거리 문맥 일관성** 학습을 더 효과적으로 수행합니다.