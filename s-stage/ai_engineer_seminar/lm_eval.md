# 자연어 처리를 위한 언어 모델의 학습과 평가

[back to super](https://github.com/jinmang2/boostcamp_ai_tech_2/tree/main/s-stage/ai_engineer_seminar)

## 언어 모델링 (Language Modeling)
- 주어진 문맥을 활용해 다음에 나타날 단어를 예측
- Bidirectional Language Modeling
    - ELMo (Embeddings from Language Models, NAACL 2018)
    - BERT (Bidirectional Encoder Representations from Transformers, NAACL 2019)

## 언어 모델의 평가
GLUE(General Language Understanding Evaluation) 벤치마크: 언어 모델 평가를 위한 영어 벤치마크

- Quora Question Pairs(`QQP`,문장유사도평가)
- Question NLI(`QNLI`,자연어추론)
- The Stanford Sentiment Treebank(`SST`,감성분석)
- The Corpus of Linguistic Acceptability(`CoLA`,언어수용성)
- Semantic Textual Similarity Benchmark`(STS-B`,문장유사도평가)
- Microsoft Research Paraphrase Corpus(`MRPC`,문장유사도평가)
- Recognizing Textual Entailment(`RTE`,자연어추론)
- SQAUD 1.1/2.0(`QA`, 질의응답)
- MultiNLI Matched(자연어추론)
- MultiNLI Mismatched(자연어추론)
- Winograd NLI(자연어추론)

본문에는 T5, BART에 대한 평가로도 사용된다고 하는데, NLU에 한정. XSum같은 요약 task는 GLUE에 없음

### 다국어 벤치마크의 등장
- FLUE(프랑스어)
- CLUE(중국어)
- IndoNLU benchmark(인도네시아)
- IndicGLUE(인도어)
- RussianSuperGLUE(러시아어)

### 한국어 자연어 이해 벤치마크(KLUE: Korean Language Understanding Evaluation)
- 개체명 인식 (Named Entity Recognition)
- 품사 태깅 및 의존 구문 분석 (POS tagging + Dependency Parsing)
- 문장 분류 (Text Classification)
- 자연어 추론 (Natural Language Inference)
- 문장 유사도 (Semantic Textual Similarity)
- 관계 추출 (Relation Extraction)
- 질의 응답 (Question & Answering)
- 목적형 대화 (Task-Oriented Dialogue)
