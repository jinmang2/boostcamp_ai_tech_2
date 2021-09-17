# Further Question

## Transformer (2)
Attention은 이름 그대로 어떤 단어의 정보를 얼마나 가져올 지 알려주는 직관적인 방법처럼 보입니다. Attention을 모델의 Output을 설명하는 데에 활용할 수 있을까요?



## Reference
- [Attention is not explanation](https://arxiv.org/pdf/1902.10186.pdf)
- [Attention is not not explanation](https://aclanthology.org/D19-1002.pdf)
- [tobigs_xai 논문 리뷰 - Attention is not explanation](https://velog.io/@tobigs_xai/5%EC%A3%BC%EC%B0%A8-Attention-is-not-explanation)
- [EMNLP-IJCNLP2019: Attention is not not explanation](https://vimeo.com/404731845)
- [jain and wallace - Attention is not Explanation](https://vimeo.com/359703968)
- [Yuval Printer - Attention is not not Explanation](https://medium.com/@yuvalpinter/attention-is-not-not-explanation-dbc25b534017)
- [Wallace - Attention is Not Not Explanation](https://medium.com/@byron.wallace/thoughts-on-attention-is-not-not-explanation-b7799c4c3b24)

## Further Reading
Discretized Integrated Gradients for Explaining Language Models
- https://arxiv.org/pdf/2108.13654v1.pdf
- Southern California
- Attention Score로 시각화하는 것의 문제제기가 많이 되어 왔음 (FAIR에 paper가 있었음)
- 때문에 아래 paper에서는 Attention-score 기반 말고 Attribution Score기반으로 하는 XAI 활용 기법제시함
  https://arxiv.org/pdf/2004.11207.pdf
- Integrated Gradients로 Attention 시각화시켜보면 +/- 방향이 안 맞는 경우가 많음
- Discretized IG라는 기법 제시!
- https://rroundtable.github.io/blog/deeplearning/xai/2020/05/05/integrated-gradient.html
- https://github.com/INK-USC/DIG
