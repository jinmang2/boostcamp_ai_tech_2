# Mission 0
pre미션에서는 Huggingface Transformer를 빠르게 리마인드 합니다.

이미 NLP, KLUE 강의를 거쳐오신 캠퍼분들께 Transformers는 익숙하실텐데요

하지만, MRC를 시작하기에 앞서 리마인드 + 놓치기 쉬운 디테일을 살펴보는 과정을 수행하고자 본 미션을 설계하였습니다.

## Contents List

### 1. Huggingface Transformers 빠르게 훑어보기

**⛔️ tokenizer 사용시 주의사항**

1. train data의 언어를 이해 할 수 있는 tokenizer인지 확인
2. 사용하고자 하는 pretrained model과 동일한 tokenizer인지 확인

  > 적절한 tokenizer를 사용하지 않을 경우 **vocab size mismatch**에러가 발생하거나 **special token이 `[unk]`**으로 처리되는 🤦🏻‍♀️대참사🤦🏻‍♂️가 벌어질 수 있음

3. 단어의 개수와 special token이 완전히 일치하는 모델은 (예를들어 klue의 roberta, bert) tokenizer를 cross로 사용 **'할 수도'** 있지만 옳은 방법은 아님
  * 첨언하자면, 공개된 영어 bert와 roberta는 tokenizer가 호환되지 않습니다. (bert vocab 28996개, roberta vocab 50265개)
  * klue bert는 동일한 기관에서 생성된 모델이므로 32000개로 총 vocab 사이즈가 동일하지만 이는 우연의 일치입니다.


**⛔️ config 사용시 주의사항**

어떤 경우에는 config를 수정하여 사용하기도 하는데, 바꾸어도 되는 config와 바꾸지 말아야 하는 config가 정해져 있습니다.

**바꾸면 안되는 config**
* Pretrained model 사용시 hidden dim등 이미 정해져 있는 모델의 아키텍쳐 세팅은 수정하면 안됩니다.
* 이를 수정해버릴 경우 에러가 발생하거나, 잘못된 방향으로 학습 될 수 있습니다.

**바꾸어도 되는 config**
* vocab의 경우 special token을 추가한다면 config를 추가한 vocab의 개수만큼 추가하여 학습해야합니다.
* downstream task를 위해 몇가지 config를 추가할 수도 있습니다. (아래에서 예시를 살펴봅시다)

**유용한 config 사용법**
- argument로 넘겨줄 경우 기존에 설정되있던 key만 추출해서 config에 넣음

```python
custom_config = {~}
config = AutoConfig.from_pretrained(model_name)
config.update(config)
```

## Huggingface Trainer

**하지만 Trainer가 '항상' 좋을까요 ?**

- **다음과 같은 모듈의 코드는 legacy가 존재할 수 밖에 없습니다.**
    :따라서 버전이 바뀔 때 마다 변동되는 사항이 많아지고 코드를 지속적으로 수정해야하는 단점이 존재합니다.
    : pytorch lightning이 대표적으로 이러한 문제를 겪고 있으며, transformers도 예외는 아닙니다.
    : 따라서 Trainer는 모든 상황에서 정답이 될 수는 없습니다

- **최대한 편리함을 이용하되, 동작 원리를 살펴보는 과정이 매우 중요합니다.**

- Trainer의 구조를 살펴보고, 내가 학습할 모델을 위한 Trainer를 만들어보는것도 좋은 방법입니다
    - Trainer에 원하는 함수 오버라이딩 하여 수정하기 (general task에 적합)
    - Custome Trainer 만들어보기 (general task가 아닌경우 유용함)

## Advanced tutorial

### token 추가하기

```python
# special token 추가하기
# 무엇이 다른가? special argument가 True인 상태로 메서드를 실행하게 된다
special_tokens_dict = {'additional_special_tokens': ['[special1]','[special2]','[special3]','[special4]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# token 추가하기
# 무엇이 다른가? special argument가 False인 상태로 메서드를 실행하게 된다
new_tokens = ['COVID', 'hospitalization']
num_added_toks = tokenizer.add_tokens(new_tokens)

# tokenizer config 수정해주기 (추후에 발생할 에러를 줄이기 위해)
config.vocab_size = len(tokenizer)

# model의 token embedding 사이즈 수정하기
model.resize_token_embeddings(len(tokenizer))
```

Q : special token을 추가할 때 항상 resize를 해주어야 하나요 ?

A : 꼭 그렇지 않습니다. 잘 만들어진 모델은 resize를 하지않고도 모델에 새로운 vocab을 추가할 수 있도록 여분의 vocab 자리를 만들어 두었습니다. 여분의 vocab 개수는 모델에 따라 다르니 확인이 필요합니다.

- 모든 모델이 dummy vocab을 고려하는 것은 아니라고 함
- SKT/KoBERT는 dummy vocab이 없고 추가 vocab을 넣을 경우에 gluonnlp를 사용하는 부분을 수정해야 한다고 함

## [CLS] 토큰 추출하기
- 모델마다 다름 (설계마다)
- 참고 : [CLS] 토큰은 정말 문장을 대표할까 ?
    - BERT의 저자 또한 [CLS]가 Sentence representation이란 것을 보장할 수 없다고 밝힘
    - https://github.com/google-research/bert/issues/164
    - 우리가 특정 task를 수행할 때 [CLS] 토큰이 당연히 문장을 대표해줄 것이라는 가정을 가지는 것은 위험함
    - 실험을 통해 어떤 토큰이 중요한 지 찾아볼 것!
    - 읽어볼만한 논문 추천 (SBERT)
        - 논문 : https://arxiv.org/pdf/1908.10084.pdf
        - **The most commonly used approach is to average the BERT output layer (known as BERT embeddings) or by using the output of the first token (the [CLS] token). As we will show, this common practice yields rather bad sentence embeddings, often worse than averaging GloVe embeddings (Pennington et al., 2014).**
        - 요약) avg나 CLS를 사용하는게 일반적이지만 이건 GLoVe의 avg보다 성능이 낮음
- 🚪✊Knock Knock을 활용하여 모델 학습 완료 알림받기

> pip install knockknock

```
from knockknock import slack_sender

webhook_url = "<webhook_url_to_your_slack_room>"
@slack_sender(webhook_url=webhook_url, channel="<your_favorite_slack_channel>")
def train_your_nicest_model(your_nicest_parameters):
    import time
    time.sleep(10000)
    return {'loss': 0.9} # Optional return value

```


command line에서 실행하기
```
knockknock slack \
    --webhook-url <webhook_url_to_your_slack_room> \
    --channel <your_favorite_slack_channel> \
    sleep 10

```
