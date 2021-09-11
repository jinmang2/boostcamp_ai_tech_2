# 1강 Introduction to NLP, Bag-of-Words
자연어 처리의 첫 시간으로 NLP에 대해 짧게 소개하고 자연어를 처리하는 가장 간단한 모델 중 하나인 Bag-of-Words를 소개합니다.

Bag-of-Words는 단어의 표현에 있어서 one-hot-encoding을 이용하며, 단어의 등장 순서를 고려하지 않는 아주 간단한 방법 중 하나입니다. 간단한 모델이지만 많은 자연어 처리 task에서 효과적으로 동작하는 알고리즘 중 하나입니다.

그리고, 이 Bag-of-Words를 이용해 문서를 분류하는 Naive Bayes Classifier에 대해서 설명합니다.

이번 강의에서는 단어를 벡터로 표현하는 방법, 문서를 벡터로 표현하는 방법에 대해 고민해보면서 강의를 들어주시면 감사하겠습니다.


[back to super](https://github.com/jinmang2/boostcamp_ai_tech_2/tree/main/u-stage/nlp)

## Academic Disciplines related to NLP

### Natural Langauage Processing
- **major conferences** : ACL, EMNLP, NAACL
- Low-level parsing
    - Tokenization, stemming
- Word and phrase level
    - NER, POS tagging, noun-phrase chunking, dependency parsing, coreference resolution
- Sentence level
    - Sentiment analysis, machine translation
- Multi-sentence and paragraph level
    - Entailment prediction, QA, dialog systems, summarization

### Text mining
- **major conferences** : KDD, The WebConf, WSDM, CIKM, ICWSM
- text와 document 데이터에서 유용한 정보와 인사이트를 추출
    - e.g., 뉴스 데이터로부터 AI 관련 트렌드 분석
- Document clustering (e.g., topic modeling)
    - e.g., 뉴스 데이터 혹은 서로 다른 주제를 군집화 혹은 그루핑
- Highly related to computational social science
    - e.g., social media 기반 사람들의 정치 성향 진화 분석

### Information Retrieval
- **major conferences** : SIGIR, WSDM, CIKM, RecSys
- Highly related to computational social science
    - e.g., Recommendation System

## Trends of NLP
Text 데이터는 단어의 수열로 이해할 수 있으며 이를 Word2Vec, GloVe 등의 방법으로 벡터화시켜서 표현했습니다. Neural Network의 발전으로 sequence 데이터 처리에 특화된 RNN-family 모델들이 NLP task의 주요 아키텍처로 자리잡았죠. 2017년 구글에서 Recurrence, Convolution 연산을 Fully Connected 연산으로 대체하는 Self-Attention을 제안했습니다. 해당 연산을 기반으로 transduction problem(seq2seq) 문제를 처리할 목적으로 제안된 Transformer Architecture의 등장으로 NLP task의 전반적인 성능은 비약적으로 높아졌습니다. 주로 기계 번역과 관련하여 성장하던 NLP task에서 다른 분야에서도 transformer가 적용되었죠. Transformer가 소개된 이후, self-attention을 쌓으므로 구현 가능한 거대 모델들이 등장했고 이들은 **self-supervised setting** 으로 특정 task에 대해 추가적인 labeling 작업을 필요로 하지 않았습니다. self-sup 방식으로 학습된 모델을 downstream task로 전이하여 학습했을 때 기존 task의 SOTA 대비 월등히 잘하는 모습을 보여줬습니다. 이러한 현상으로 NLP는 제한된 GPU resource에선 연구가 힘든 상황이 되어버렸죠...

## Bag-of-Words
- Step-1. Constructing the vocabulary containing unique words
- Step-2. Encoding unique words to one-hot vectors
    - 각 단어 pair별 Euclidean Distance는 $\sqrt{2}$
    - 각 단어 pair별 Cosine Similarity는 $0$
- A sentence/document can be represented as the sum of one-hot vectors

## NaiveBayes Classifier for Document Classification

Bayes' Rule Applied to Documents and Classes

- MAP: Maximum a posteriori

$$\begin{array}{lll}
C_{MAP}&=\argmax_{c\in C}P(c|d)\\
&=\argmax_{c\in C}\cfrac{P(d|c)P(c)}{P(d)}\\
&=\argmax_{c\in C}P(d|c)P(c)
\end{array}$$

$$P(d|c)P(c)=P(w_1,w_2,\dots,w_n|c)P(c)\rightarrow P(c)\prod_{w_j\in W}P(w_j|c)$$
