
### 🔹 Post-LN (논문 원형 2017)

`x → Sublayer → Add → LayerNorm`

### 🔹 Pre-LN (GPT / BERT 이후 표준)

`x → LayerNorm → Sublayer → Add`

✅ **현대 대형 모델(GPT-3+, LLaMA, PaLM 등)은 거의 전부 Pre-LN**

- 이유:
    
    - 깊어질수록 학습 훨씬 안정
    - Gradient flow가 좋음