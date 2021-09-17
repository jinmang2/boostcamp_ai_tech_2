# Further Question

## Self-supervised Pre-training Models
Attention은 이름 그대로 어떤 단어의 정보를 얼마나 가져올 지 알려주는 직관적인 방법처럼 보입니다. Attention을 모델의 Output을 설명하는 데에 활용할 수 있을까요?

## Answer
- BERT의 Masked Language Model의 단점은 무엇이 있을까요? 사람이 실제로 언어를 배우는 방식과의 차이를 생각해보며 떠올려봅시다
    - Mask를 씌우는 방식에서 문제가 있다.
    - 여러개의 빈칸이 있을 경우 사람은 Mask 간의 관계도 고려해서 파악
    - 일상 Task에서도 Mask가 뚫려있지않음, 사전학습과 실제 Task간의 불일치
    - 외부데이터(배경지식, 상식)를 활용시 문제가 있을 수 있다.
- CutMix와의 차이?
    - 이건 복원하는 방식은 아님
    - DEiT 가 MLM과 비슷한 방식을 씀
