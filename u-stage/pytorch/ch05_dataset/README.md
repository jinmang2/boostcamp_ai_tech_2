# 4강 Autograd and Optimizer
- 이번 강의에선 Pytorch Dataset, Dataloader를 사용하는 방법을 학습합니다. 데이터 입력 형태를 정의하는 Dataset 클래스를 학습하여 Image, Video, Text 등에 따른 Custom Data를 PyTorch에 사용할 수 있도록 학습하고, DataLoader를 통해 네트워크에 Batch 단위로 데이터를 로딩하는 방법을 배웁니다.

- 본 강의에서는 NotMNIST 데이터를 통해 직접 구현해보는 시간을 가집니다.

[back to super](https://github.com/jinmang2/boostcamp_ai_tech_2/tree/main/u-stage/pytorch)

## 모델에 데이터를 먹이는 방법?

![img](../../../assets/img/u-stage/pytorch_05_01.PNG)

## Dataset 클래스
- 데이터 입력 형태를 정의하는 클래스
- 데이터를 입력하는 방식의 표준화
- Image, Text, Audio 등에 따른 다른 입력 정의

### 유의사항
- 데이터 형태에 따라 각 함수를 다르게 정의
- 모든 것을 데이터 생성 시점에 처리할 필요는 없다!
- 데이터 셋에 대한 표준화된 처리방법 제공 필요
  - 후속 연구자 또는 동료를 위해...!

## DataLoader 클래스
- Data의 Batch를 생성해주는 클래스
- 학습 직전 (GPU feed전) 데이터의 변환을 책임
- Tensor로 변환 + Batch 처리가 메인 업무
- 병렬적인 데이터 전처리 코드의 고민 필요

```python
class DataCollator:
    def __init__(self, tokenizer, max_length=510):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        return self.tokenizer(
            text=[x["source"] for x in batch],
            text_pair=[x["target"] for x in batch],
            src_langs=[x["src_lang"] for x in batch],
            tgt_langs=[x["tgt_lang"] for x in batch],
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )
```

## Case study
- 데이터 다운로드부터 loader까지 직접 구현해보기!
- NotMNIST
- http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html
