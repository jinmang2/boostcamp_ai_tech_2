# Further Question

## Beam Search and BLEU Score
- BLEU score가 번역 문장 평가에 있어서 갖는 단점은 무엇이 있을까요?

A) Human Evaluation과 다르다
- 사람은 n-gram을 카운트해서 번역이 잘 되었음을 평가하지 않는다.
- 다른 의미적인 표현이 있을 수 있다.(의미론적 문장 유사관계)
- 높은 BLEU Score가 인간이 보기에 높은 평가를 받는 문장이 아닐 수 있다.
- HyperCLOVA 발표 & BLEURT예측 문장의 벡터와 타겟 문장의 벡터의 유사도를 이용해 생성된 문장의 품질을 평가하는 방법
- [BLEURT](https://greeksharifa.github.io/machine_learning/2021/01/13/BLEURT-Learning-Robust-Metrics-for-Text-Generation/)
- [BARTScore](https://arxiv.org/abs/2106.11520)
