# AI + ML과 Quant Trading

[back to super](https://github.com/jinmang2/boostcamp_ai_tech_2/tree/main/s-stage/ai_engineer_seminar)

## 뭐 하는 사람인가?
- HFT (High Frequency Trading) Quant Developer / Portfolio manager
    - 뉴욕 Two Sigma Securities (2014~2020)
    - 뉴욕 Tower Research Capital (2021~)
- 컴퓨터 과학 및 개발자 백그라운드
    - 네이버 (2006~2008)
    - algospot.com (2007~)
    - 알고리즘 문제 해결 전략 (2011)

## Disclaimer
- 트레이딩 업계는 `지적재산권`을 매우 중시 (즉, 비밀스러움)
- 실제 딥러닝 리서치가 활용되는 예는 아직 드움
- 재미삼아 들어주세요 ^^

## What is trading?
- Trading은 투자(Investment) 대비 단기간
- 길어야 ~3일 정도의 보유기간

### Quant Trading
- Quantitative Trading
- 모델 기반 혹은 데이터 기반
    - 가격이 특정 수학적 성질을 가지는가?
    - 시장의 과거 데이터에서 분포를 추정
- Automated/system/algorithmic trading

수 많은 종류의 전략이 존재함
- 포지션을 얼마나 오래 유지하는가?
- 어떤 상품군을 거래하는가?
- 100% 자동화되었는가? 혹은 트레이더의 주관이 들어가는가?
- 주문 집햅(trade execution) vs 자체 수익
- 어디에서 엣지가 오는가? (시장의 특성을 이용? 훌륭한 통계적 모델?)

### e.g.
- Arbitrage(차익거래)
    - 90% 속도 + 10% 알파
- Market Making
    - 50% 속도 + 50% 알파
- Statistical Arbitrage
    - 10% 속도 + 90% 알파
    - 데이터 기반 접근이 필수적

### Players
- 퀀트 헤지펀드 / 로보 어드바이저
- 프랍 트레이딩 (자기 자본 거래)
- 금융위기 이후 규제 변경으로 은행들은 자기자본 거래를 하지 않음

### Deep Learning is all you need?
Nope! 선형회귀 #1~#10, 머신러닝 조금, 딥러닝 조금 + Porfolio Optimizer (model-based) + risk model

## 이게 정말 되나요?
효율적 시장 가설(Eugene Fama): 가격은 상품에 대한 모든 정보를 포함하고 있기 때문에 장기적으로 초과수익을 얻을 수는 없다

안된다는 증거와 썰
- 액티브 펀드 매니저들과 시장 인덱스의 퍼포먼스 비교
- 액티브 펀드 매니저와 원숭이의 대결
- Long Term Capital Management (1994~1998)

### 미래가 예측 가능한 이유들 (주장)
- 포지션이 큰 참가자들은 움직이는 데 오래 걸림
- 큰 가격 변화가 있을 때 군중 심리가 나타난다
- 프로페셔널 참가자들은 리스크를 줄이는 합리적 행동을 한다
- 새로운 정보(뉴스, 공시 정보, 매출 및 펀더멘탈)가 시장에 반영되기까지는 시간이 걸린다
- 기술적ㅇ니 문제들: 거래소 / 종목의 특성, 특정 규칙에 따라 움직여야 하는 참가자들,...
- 거래량이 많은 상품이나 거래소가 가격 발견 과정을 선도한다.

상품에 대한 새로운 정보가 가격에 포함되기 위해서는 누군가 거래를 해야 한다.

### 성공 기준이 우리의 직관과 다르다
- 엄청나게 많은 작은 예측들
- 엄청나게 많은 forecast 알고리즘
- High Frequency Trading 전략의 평균 수익은 거래량의 대략 0.01% (1 basis point)!
    - 실제 수익은 훨씬 크거나 훨씬 작지만, 평균 내면 아주 작지만 이익
    - 이것으로 연수익 100%+를 내려면 엄청나게 많은 양을 꾸준히 거래해야 한다

### 어... 나도 가능?
안됨^^ ㅎ...

(뇌피셜) Google Finance, DART, 트위터, 뉴스 크롤링 + 캔들스틱 + MACD, 볼린저 밴드!

실패...

## 왜 딥러닝을 안하나요?
- 시장을 예측하는 것은 너무 어려움
    - 볼 수 있는 정보도 한정적, risk, shifting
- 시장은 계속 변함
    - 접근 방법 고도화, 규제 조건 바뀜, 새로운 상품군 출현

## 어떤 리서치를 하는가?
오렌지 + 쥬서기 = 쥬스
- 오렌지: 새로운 정보
- 쥬서기: 새로운 정보에서 어떻게 요점만 뽑아낼 것인가?
- 쥬스: 알파(초과 수익)

### 리서치 과정에서 흔히 주의해야 하는 것
- 프로덕션 시스템 vs 백테스트 시스템 차이
- 마켓 임팩트
- 데이터 스누핑
