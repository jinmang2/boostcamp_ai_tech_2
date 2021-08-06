# Further Question

## Gradient Descent - Spicy
- d차원 벡터에 대한 그레디언트 벡터 구하는 계산을 직접 해보기

$$\begin{array}{ll}
\partial{\beta_k}\mathbb{E}[\lvert\lvert y-X\beta \rvert\rvert_2]&=\partial_{\beta_k}\bigg\{ \cfrac{1}{n}\sum_{i=1}^{n}{\bigg(y_i-\sum_{j=1}^{d}X_{ij}\beta_j\bigg)}^2 \bigg\}^{1/2}\\
&=-\cfrac{X_{\cdot k}^{\intercal}(y-X\beta)}{n\lvert\lvert y-X\beta\rvert\rvert_2}
\end{array}$$

`Want to show` $$\partial{\beta_k}\mathbb{E}[\lvert\lvert y-X\beta \rvert\rvert_2]=-\cfrac{X_{\cdot k}^{\intercal}(y-X\beta)}{n\lvert\lvert y-X\beta\rvert\rvert_2}$$

`proof`
$$\begin{array}{lllll}
\partial{\beta_k}\mathbb{E}[\lvert\lvert y-X\beta \rvert\rvert_2]&=\partial_{\beta_k}\bigg\{ \cfrac{1}{n}\sum_{i=1}^{n}{\big(y_i-\sum_{j=1}^{d}X_{ij}\beta_j\big)}^2 \bigg\}^{1/2}\\
&=\cfrac{1}{2}{\bigg(\cfrac{1}{n}\sum_{i=1}^{n}{\big(y_i-\sum_{j=1}^{d}X_{ij}\beta_j\big)}^2\bigg)}^{-1/2}\cdot\cfrac{\partial}{\partial \beta_k}\bigg(\cfrac{1}{n}\sum_{i=1}^{n}{\big(y_i-\sum_{j=1}^{d}X_{ij}\beta_j\big)}^2\bigg)\\
&=\cfrac{1}{2}\cdot\cfrac{1}{\lvert\lvert y-X\beta \rvert\rvert_2}\cdot\cfrac{\partial}{\partial \beta_k}\bigg(\cfrac{1}{n}\sum_{i=1}^{n}{\big(y_i-\sum_{j=1}^{d}X_{ij}\beta_j\big)}^2\bigg)\\
&=\cfrac{1}{2}\cdot\cfrac{1}{\lvert\lvert y-X\beta \rvert\rvert_2}\cdot\cfrac{2}{n}\sum_{i=1}^{n}{\big(y_i-\sum_{j=1}^{d}X_{ij}\beta_j\big)}\cdot\cfrac{\partial}{\partial \beta_k}{\big(y_i-\sum_{j=1}^{d}X_{ij}\beta_j\big)}\\
&=\cfrac{1}{n\lvert\lvert y-X\beta \rvert\rvert_2}\cdot-X_{\cdot k}^\intercal(y-X\beta)\\
&=-\cfrac{X_{\cdot k}^\intercal(y-X\beta)}{n\lvert\lvert y-X\beta \rvert\rvert_2}\\
\end{array}$$

- 중간에 5번째 줄이 너무 갑작스럽게 change됐는데,
- (1xn)*(nx1) 꼴로 연산이 들어가기 때문에 행렬꼴로 바꿔주면서
- 모양이 change된 것.
- 만일 계수에 대해 편미분이 아니라 bias에 대해 들어간다면 저기에 있는 $X_{\cdot k}^\intercal$부분이 사라질 것.
