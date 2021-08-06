# Further Question

## Probability
- 몬테카를로 방법을 활용하여 원주율에 대한 근삿값을 어떻게 구할 수 있을까요?

```python
import math
import random


in_ball = 0
total = 1000000

for i in range(total):
    x = random.random()
    y = random.random()
    if math.sqrt(x ** 2 + y ** 2) < 1.0:
        in_ball += 1

pi = float((in_ball) / total) * 4
pi
```
