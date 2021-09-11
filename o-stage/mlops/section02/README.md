# Section 2. 코드 품질, 데이터 검증, 모델 분석
- 출처: https://www.inflearn.com/course/머신러닝-엔지니어-실무/

[back to super](https://github.com/jinmang2/boostcamp_ai_tech_2/tree/main/o-stage/mlops)

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#리서치-코드-품질-관리-자동화">리서치 코드 품질 관리 자동화</a>
      <ul>
        <li><a href="#why-did-i-apply-for-this-project?">리서치 코드 품질 문제</a>
        <ul>
          <li><a href="#1-코드-중복">코드 중복</a></li>
          <li><a href="#2-너무-많은-전역-변수">너무 많은 전역 변수</a></li>
          <li><a href="#3-너무-긴-코드">너무 긴 코드</a></li>
          <li><a href="#4-이상하게-꼬여있는-import">이상하게 꼬여있는 import</a></li>
          <li><a href="#5-명확하지-않은-변수명">명확하지 않은 변수명</a></li>
        </ul>
        </li>
        <li><a href="#린트-유닛-테스트">린트, 유닛 테스트</a>
        <ul>
          <li><a href="#python-black">python black</a></li>
          <li><a href="#linter-flake8">Linter - flake8</a></li>
          <li><a href="#python-타입-체크-mypy">Python 타입 체크 - mypy</a></li>
        </ul>
        </li>
        <li><a href="#지속적-통합">지속적 통합</a></li>
        <ul>
          <li><a href="#github-actions">Github Actions</a></li>
        </ul>
      </ul>
    </li>
    <li><a href="#데이터-검증-tensorflow-data-validation">데이터 검증 - Tensorflow Data Validation</a>
    <li><a href="#머신러닝-모델-분석-what-if-tool">머신러닝 모델 분석 What if tool</a></li>
  </ol>
</details>

## 리서치 코드 품질 관리 자동화

SW Engineer vs Researcher의 코드 관리. 어떻게 가이드 라인을 작성할 것인지?

- 리서치 코드 품질 문제
- 린트, 유닛 테스트
- 지속적 통합 (Contiguous Integration)

### 리서치 코드 품질 문제

코드 관리를 하지 않은 리서치 코드는 보통 아래의 문제점을 가진다.

1. 리서치 코드는 각자의 개인 컴퓨터에 저장
2. 코드는 매번 복사 붙여넣기로 개발, 코드 중복이 많음
3. 연구 결과는 재연이 불가능
4. 수 많은 코드 악취가 남아있음

```
"깨진 유리창의 법칙 (Broken Windows Theory)"

만일 한 건물의 유리창이 깨어진 채로 방치되어 있다면,
곧 다른 유리창들도 깨어질 것.
```
깨진 유리창의 법칙으로 코드 품질 미관리는 엄청난 비효율로 이어지게 되고 이를 관리해야 `pipeline`화가 가능해진다.

구체적으로 어떤 문제가 있을까?

<br/>
<div align="right">
    <b><a href="#section-2-코드-품질-데이터-검증-모델-분석">↥ back to top</a></b>
</div>
<br/>

#### 1. 코드 중복
- 보통 git에서 clone해오거나 복사해온 리서치 코드가 대다수임
- 만일 해당 코드에 취약점이 있을 경우, 사본을 알지 못하는 경우 위 취약점은 해당 코드를 가져올 때마다 남아있게 된다.
- 중복이 될만한 코드(예를 들어 전처리)는 재사용 가능하게 추상화하여 관리하는 것이 필수

#### 2. 너무 많은 전역 변수
- 가능한 환경 값은 환경 변수 활용
- 함수에 명시적으로 파라미터를 전달하고 받아오는 방식으로 고쳐야 함

#### 3. 너무 긴 코드
- 디버깅하기 불편, 함수와 클래스 구분이 명확하지 않게되는 경향이 존재
- 뤼이드의 경우, 정적 분석 Tool을 사용하여 코드를 넣으면 code snippet을 알아서 체크해서 Issue를 만들어줘서 잘못된 artifact들이 있으면 merge가 안되게 설정했다고 함.

[참고: code snippet?](https://soeasyenglish.tistory.com/entry/Q-code-snippet-%EC%BD%94%EB%93%9C-%EC%8A%A4%EB%8B%88%ED%95%8F-%EB%AC%B4%EC%8A%A8-%EB%9C%BB)
- snippet이란 작은 조각이라는 뜻!
- Code Snippet이라는 것은 코드의 일부분만 발췌한 것을 의미

#### 4. 이상하게 꼬여있는 import
- relative import를 여기저기서 사용하게 되면 서로 참조 관계가 얽혀서 나중에는 디버깅이 어려운 수준까지 가게 된다.
- `PYTHONPATH` 환경변수를 활용하여 현재 시작 지점을 명확하게 하고 absolute import를 사용하자

[참고: import order convention](https://www.python.org/dev/peps/pep-0008/#imports)
1. Standard library imports
2. Related third party imports
3. Local

아래처럼 absolute import로 하는 것을 추천!

```python
import mypkg.sibling
from mypkg import sibling
from mypkg.sibling import example
```

#### 5. 명확하지 않은 변수명
- 너무 축약어를 쓰다보면 나밖에 모르는 코드가 된다

<br/>
<div align="right">
    <b><a href="#section-2-코드-품질-데이터-검증-모델-분석">↥ back to top</a></b>
</div>
<br/>


### 린트, 유닛 테스트
- [Python 코드 스타일 툴 (Pycharm 환경 Flake8, Black...)](https://medium.com/daehyun-baek/python-%EC%BD%94%EB%93%9C-%EC%8A%A4%ED%83%80%EC%9D%BC-%ED%88%B4-pycharm-%ED%99%98%EA%B2%BD-flake8-black-4adba134696a)

#### python black
- [Black이란?](https://www.daleseo.com/python-black/)
- https://github.com/psf/black#editor-integration
- Formatter로 이용하여 코드를 정리
- 최근 python 커뮤니티에서 가장 널리 쓰이고 있는 코드 포멧터
- 정해놓은 특정 포멧팅 규칙을 그대로 따라가야 한다.

아래는 cli 환경에서 직접 수행
```
pip install black

black --check main.py
```

아래는 vscode 및 git hook 설정 방식
```
# .vscode/settings.json
{
  "editor.formatOnSave": true,
  "python.formatting.provider": "black"
}
```

```python
# install git hook
$ pip install pre-commit

# .pre-commit-config.yaml 파일 생성 후 다음과 같이 설정 추가
repos:
  - repo: https://github.com/psf/black
    rev: stable
    hooks:
      - id: black

# pre-commit 커맨드를 싱행하여 git hook 스크립트 설치
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit

# 코드 에디터 자동 포메팅 해제
# new formatting
$ git commit
black....................................................................Failed
- hook id: black
- files were modified by this hook

reformatted /Users/dale/learn/learn-python/main.py
All done! ✨ 🍰 ✨
1 file reformatted, 9 files left unchanged.
```

**python indent**
- PEP 8에 따라 공백 4칸이 원칙
- Google python guide-line 또한 공백 4칸
- 첫 번째 줄에 파라미터가 있다면, 파라미터가 시작되는 부분에 보기 좋게 맞춘다.

```python
foo = long_function_name(var_one, var_two,
                         var_three, var_four)
```

- 이 코드처럼 첫 번째 줄에 파라미터가 없다면, 공백 4칸 인덴트를 한번 더 추가하여 다른 행과 구분

```python
def long_function_name(
    var_one, var_two, var_three,
    var_four):
    print(var_one)
```

**Naming Convention**
- 자바와 달리 각 단어를 밑줄로 구분하여 표기하는 snake case를 따름
- pythonic way에 굉장한 자부심, java style coding을 지양함
    - java는 camel case를 따름

<br/>
<div align="right">
    <b><a href="#section-2-코드-품질-데이터-검증-모델-분석">↥ back to top</a></b>
</div>
<br/>

#### Linter - flake8

[What is Flake8 and why we should use it?](https://medium.com/python-pandemonium/what-is-flake8-and-why-we-should-use-it-b89bd78073f2)
- Lint란, symantec discrepancies를 위한 source code checking 정적 분석 툴
- Linting이란, 기본 quality tool을 코드에 수행하는 것.
- 왜 Linting이 중요하고 꼭 수행해야 하는가?
    - Linting은 더 좋은 코드를 작성하는 나은 개발자로 만들어 준다
    - syntax errors, typos, bad formatting, incorrect styling 등을 방지하는데 도움을 준다
    - 개발 시간을 절약해준다
    - 팀 단위로 움직일 때 리뷰잉 시간을 줄여준다
    - 사용하기 쉽다
    - Lint-like tool들은 설치가 용이하다
    - 무료다 :)

Flake8이란?
- python code linter tool
- PEP8, pyflakes 및 순환 복잡성을 확인하는 wrapper
- false positive 비율이 낮음

```python
python3.x -m flake8

flake8 -help

flake8 path/to/your_project/ # check the entire project repo
```

#### Python 타입 체크 - mypy
- [Mypy - DaleSeo](https://www.daleseo.com/python-mypy/)
- python은 동적 언어지만 static 타입 체크가 필요할 때가 있음
- application의 규모가 커지면 다이나믹함이 치명적인 버그로 작용함.
- type annotation은 python3.5에서 추가됨
- 이 표준에 따르면 static type checker를 사용하여 코드를 실행하지 않고도 타입 에러를 찾아낼 수 있음

```python
# mypy 설치
$ pip install mypy
$ mypy our_file.py # 파일 검사
$ mypy our_directory # 디렉토리 검사
```

- 파이썬에서 기본적으로 지원하는 자료형은 아래와 같음

```python
Text Type : str
Numeric Types : int, float, complex
Sequence Types : list, tuple, range
Mapping Type : dict
Set Types : set, frozenset
Boolean Type : bool
Binary Types : bytes, bytearray, memoryview
```

- 아니면 내장 라이브러리인 `typing`을 이용해서 어노테이팅할 수 있다.


<br/>
<div align="right">
    <b><a href="#section-2-코드-품질-데이터-검증-모델-분석">↥ back to top</a></b>
</div>
<br/>

### 지속적 통합

Software Engineering에서 지속적 통합(Continuous Integration, CI)는 **지속적으로 퀄리티 컨트롤을 적용하는 프로세스를 실행하는 것**
- 작은 단위의 작업, 빈번한 통합

CI는 모든 개발을 완료한 뒤에 quality control을 적용하는 고전적인 방법을 대체하는 방법

SW의 질적 향상 및 배포 시간을 줄이는데 초점!

#### Github Actions
[실습 결과]()

**보이 스카우트 규칙**
- 떠날 때는 찾을 때보다 캠프장을 더욱 깨끗이 할 것!

<br/>
<div align="right">
    <b><a href="#section-2-코드-품질-데이터-검증-모델-분석">↥ back to top</a></b>
</div>
<br/>

## 데이터 검증 - Tensorflow Data Validation

<br/>
<div align="right">
    <b><a href="#section-2-코드-품질-데이터-검증-모델-분석">↥ back to top</a></b>
</div>
<br/>

## 머신러닝 모델 분석 What if tool

<br/>
<div align="right">
    <b><a href="#section-2-코드-품질-데이터-검증-모델-분석">↥ back to top</a></b>
</div>
<br/>
