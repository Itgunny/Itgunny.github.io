### Linear Regresson의 cost  최소화 알고리즘의 원리 설명\

공식은 다음과 같다. 

$H(x) = Wx + b$

$cost(W, b) = \frac{1}{m} \sum_{i=1}^{m}(H(x^{i})-y^{i})^2​$

### Simplified hypothesis

간단히 하기 위해서 b를 없애고 생각해보려고 한다.(b = 0)

$H(x) = Wx$

$cost(W) = \frac{1}{m} \sum_{i=1}^{m}(H(x^{i})-y^{i})^2​$

아래의 표를 예로 들때 

| x    | y    |
| ---- | ---- |
| 1    | 1    |
| 2    | 2    |
| 3    | 3    |

1) $W = 1, cost(W) = ?​$

$(1\times1 - 1)^2 + (1\times2 - 2)^2 + (1\times3 - 3) ^2  = 0​$가 0이되므로 $cost(W) = 0​$이 된다.

2) $W = 0, cost(W) = 4.67​$

$\frac{1}{3}(0\times1 - 1)^2 + (0\times2 - 2)^2 + (0\times3 - 3) ^2  = 4.67​$

3) $W = 0, cost(W) = 4.67$

$\frac{1}{3}(2\times1 - 1)^2 + (2\times2 - 2)^2 + (2\times3 - 3) ^2  = 4.67$ 

이와 같은 점들을 이어보면 2차함수와 같은 그래프가 표시 된다. 우리는 코스트가 최소화 되는 점을 찾고 있다. W = 1일때 코스트가 0으로 최소화 된다. 

이를 기계적으로 찾는 알고리즘이 Gradient descent algorithm(경사 하강 알고리즘) : cost function을 최소화 하는데 사용된다. 

또한 cost(w1, w2 ...) 많은 w가 있어도 최소화 할 수 있는 알고리즘이다. 

기울기를 따라서 한발자국씩 움직이면서 기울기가 최소화되는 지점을 찾는 알고리즘이라고 간략하게 설명할 수 있다. 

### How it works?

1. Start with initial guesses

- Start at 0, 0(or any other value)
- Keeping changing W and b a little bit to try and reduce cost(W, b)

2. Each time you change the parameters, you select the gradient which reduces cost(W, b) the most possible
3. Repeat
4. Do so until you converge to a local minimum
5. Has an interesting property

- Where you start can determine which minimum you end up

### Formal definition

결국 기울기라는 것은 미분을 의미 한다. 수식을 간단하기 위해서 2m으로 만들어 주었다. 

$cost(W) = \frac{1}{m} \sum_{i=1}^{m}(H(x^{i})-y^{i})^2​$

$cost(W) = \frac{1}{2m} \sum_{i=1}^{m}(H(x^{i})-y^{i})^2​$

-> $cost(W) = \frac{1}{m} \sum_{i=1}^{m}(H(x^{i})-y^{i})^2​$

$W := W - \alpha\frac{\partial}{\partial W}cost(W)​$ 

-> $W := W - \alpha\frac{\partial}{\partial W}\frac{1}{2m} \sum_{i=1}^{m}(W(x^{i})-y^{i})^2​$

-> $W := W - \alpha\frac{1}{2m}\sum_{i=1}^{m}2(W(x^{i})-y^{i})*x^i​$

-> $W := W - \alpha\frac{1}{m}\sum_{i=1}^{m}(W(x^{i})-y^{i})*x^i$

한점에서 기울기를 구해서 W를 미분한만큼 이동해 주는 공식이라고 할 수 있다.

### Convex function

$cost(W) = \frac{1}{m} \sum_{i=1}^{m}(H(x^{i})-y^{i})^2$

결국에 W, b의 minimal cost 지점이 동일하면 Convex function이라고 부른다. 항상 GDA를 사용하기 위하여는 Convex인 것을 확인하여야 한다.





