# 3강 파이썬 기초 문법 2

[back to super](https://github.com/jinmang2/boostcamp_ai_tech_2/tree/main/u-stage/python_basic)

## 3.1 Python Data Structure
- 특징이 있는 정보를 어떻게 저장하는 것이 좋을까?
    - stack & queue
    - tuple & set
    - dictionary
    - collection module
- `stack, queue` :  파이썬 리스트로 구현 pop()의 index만 다릅니다.

- `tuple` : 값 변경 불가한 리스트입니다.

- `set` : 중복 안되는 자료형, 다양한 집합 연산

- `dictionary` : 다른 언어에서는 hash map {key: value}

  - command analyzer
  - word counter에 활용

- `deque` : stack, queue를 지원하고 linked list의 특성

  - 리스트보다 효율적인 자료구조
  - 처리 속도 향상 (=why?)
  - linked_list의 insert 시간 복잡도는 O(1)이고, 리스트의 insert 시간 복잡도는 O(n)입니다. 하지만 append와 pop의 시간 복잡도는 둘 다 O(1)이기 때문에 deque이 무조건 빠르다고 볼 수는 없습니다.

- `ordereddict` : 순서를 보장한 사전방식

- `defaultdict`  : 기본값을 지정해서 신규값 조회 때 일어나는 에러방지

- `Counter` : data element의 개수를 dict 형태로 반환, set 연산 지원

- `namedtuple` : data 변수를 지정해서 사용(깔끔)

## 3.2 Pythonic code
- `Split` : 기준값으로 나눠서 리스트로 변환

- `join` : string이 원소인 리스트를 합쳐서 string으로 변환

- `list comprehension` : 리스트를 간단히 만드는 방법으로 for+append보다 속도가 빠르다고 합니다.(why?)
    - append라는 리스트 객체의 함수를 실행하기 위해서는 함수를 호출해야하는데, List comprehension은 Function Call 횟수를 줄였기 때문입니다.

- `enumerate` : list element에 번호 붙여서 추출

- `zip` : 두 개의 list 값을 병렬적으로 추출함

- `lambda` : lambda x, y : x+y

- `map` : 실행시점의 값 생성하여 메모리 효율적으로 원소마다 함수 적용

- `reduce` : 리스트에 똑같은 함수 적용 후 통합

- `iterable object`
    - 리스트와 같이 Sequence형 자료형에서 데이터를 순서대로 추출, iter(), next()로 가능

- `generator` : yield로 한번에 하나의 element 반환
    - Pytorch dataloader, keras fit_generator가 여기에 해당

- `function passing arugments` 튜플타입, 딕셔너리 타입

![img](../../../assets/img/u-stage/passing_arguments.PNG)
