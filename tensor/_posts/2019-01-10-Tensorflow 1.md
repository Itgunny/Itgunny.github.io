## 텐서플로우 프로그래밍 개념

### 1. 학습목표 : 아래의 개념에 중점을 두고 텐서플로우 프로그래밍 모델의 기본 사항을 배웁니다. 

​     1) 텐서

​     2) 연산

​     3) 그래프

​     4) 세션 

​    기본 그래프를 만드는 간단한 텐서플로우 프로그램 만들기 및 그래프를 실행하는 세션

---

### 2. 개념 개요

텐서플로우는 임의의 차원을 갖는 배열들을 뜻하는 텐서에서 그 이름이 유래되었다. 텐서 플로우를 사용하면 차원수가 아주 높은 텐서를 조작할 수 있다. 하지만, 대부분은 다음과 같은 저차원 텐서 중 하나 이상을 사용하여 작업하게 됩니다. 

스칼라는 0-d 배열(0번째 텐서)입니다. 예 : 'Howdy' 또는 5

벡터는 1-d 배열(1번째 텐서)입니다.  예: [2, 3, 5, 7, 11] 또는 [5] 

행렬은 2-d 배열(2번째 텐서) 입니다. 예: [(3.1, 8.2, 5.9)(4.3, -2.7, 6.5)]

텐서 플로우 연산은 텐서를 만들고 없애고 조작합니다. 일반적인 텐서플로우 프로그램에서 대부분의 코드 행은 연산입니다. 

텐서플로우는 그래프(또는 산출 그래프나 데이터 플로우 그래프)는 그래프 데이터 구조 입니다. 많은 텐서플로우 프로그램은 하나의 그래프로 구성되어 있지만, 텐서플로우 프로그램은 여러 그래프를 만들 수 도 있습니다. 그래프의 노드는 연산이고; 그래프의 엣지는 텐서입니다. 텐서는 그래프를 따라 흐르고, 각 노드에서 연산에 의해 조작됩니다. 한 연산의 출력 텐서는 보통 다음 연산의 입력 텐서가 됩니다. 텐서플로우는 레이지 실행 모델을 구현하는데, 이는 연결된 노드의 필요에 따라 필요할 때만 노드가 계산된다는 의미입니다. 

텐서는 그래프에서 상수 또는 변수로 저장될 수 있습니다. 상수와 변수가 그래프에선 다른 연산이 될 수 있습니다. 상수는 항상 같은 텐서 값을 반환하는 연산이고, 변수는 할당된 텐서를 반환합니다. 

상수를 정의 하려면 tf.constant 연산자를 사용하여 그값을 전달합니다. 예를 들면 다음과 같습니다. 

```python
x = tf.constant([5.2]) # constant : 상수 텐서 선언
```

유사하게 다음과 같은 변수를 만들 수 있습니다. 

```python
y = tf.Variable([5]) # Variable : 변수 텐서 선언
```

또는 변수를 먼저 만든 다음, 다음과 같은 값을 할당할 수 있습니다. 참고로 항상 기본 값을 지정해야 합니다. 

```python
y = tf.Variable([0]) # Variable : 기본값 지정 후 선언
y = y.assign([5]) # 할당
```

일부 상수 또는 변수를 정의하면 tf.add와 같은 연산과 병합할 수 있습니다. tf.add 연산을 평가할 때 tf.constant 또는 tf.Variable 연산을 호출하여 값을 얻은 다음 그 값의 합으로 새 텐서를 반환합니다. 

그래프는 반드시 텐서플로우 세션 내에서 실행되어야 합니다. 세션은 다음을 실행하는 그래프의 상태를 가집니다. 

```python
with tf.Session as sess:
	initialization = tf.global_variables_initializer()
	print(y.eval())
```

tf.Variable을 사용할 때 위에서와 같이 세션 시작시 tf.global_variable_initializer를 호출하여 명시적으로 초기화해야 합니다. 

---

### 3. 요약

텐서플로우 프로그래밍은 기본적으로 두 단계 과정입니다. 

1) 상수, 변수, 연산을 그래프로 결합합니다. 

2) 이 상수, 변수, 연산을 세션 내에서 평가합니다.

---

### 4. 간단한 텐서플로우 프로그램 만들기

두개의 상수를 더하는 간단한 텐서플로우 프로그램 코딩 방법을 알아보겠습니다. 

#### 4. 1 Import 명령문 제공

거의 모든 Python 프로그램에서와 마찬가지로 먼저 몇가지 import 명령문을 지정하는 것으로 시작합니다. 텐서플로우 프로그램을 실행하는 데 필요한 import 명령문 집합은 물론 프로그램에서 액세스 하는 기능에 따라 달라집니다. 최소한 모든 텐서플로우 프로그램에서 import tensorflow 명령문을 제공해야 합니다. 

```python
import tensorflow as tf
```

다른 일반적인 import  명령문은 다음과 같습니다. 

```Python
import matplotlib.pyplot as plt # 데이터세트 시각화
import numpy as np # 저수준 숫자 Python 라이브러리
import pandas as pd # 고수준 숫자 Python 라이브러리
```

텐서플로우는 기본 그래프를 제공합니다. 하지만 명시적으로 나만의 그래프를 만들어 추적 상태를 촉진하는 것이 좋습니다. 예를 들어 각 셀에서 다른 그래프로 작업하고 싶을 수 있기 때문입니다. 

```python
from __future__ import print_function
import tensorflow as tf

# Create a graph
g = tf.Graph()

# Establish the graph as the "default graph.
with g.as_default():

# Assemble a graph consisting of the following three operations:
# * Two tf.constant operations to create the operands.
# * One tf.add operation to add the two operands.
x = tf.constant(8, name="x_const")
y = tf.constant(5, name="y_const")
sum = tf.add(x, y, name="x_y_sum")

# Now create a session.
# The session will run the default graph.
with tf.Session() as sess:
	print(sum.eval())
```



#### 4. 2 실습 세 번째 피연산자 도입

위의 코드 목록을 수정하여 두 개 대신 세 개의 정수를 추가합니다. 

1) 세 번째 스칼라 정수 상수 z를 정의하고 값 4를 할당합니다. 

2) 합계에 z를 더해서 새 합계를 도출합니다. 

3) 수정된 코드 블록을 다시 실행합니다.

```python
from __future__ import print_function

import tensorflow as tf

# Create a graph
g = tf.Graph()

# Establish our graph as the "default" graph.
with g.as_default();

# Assemble a graph consisting of three operations.
# (Creating a tensor is an operation.)
x = tf.constant(8, name="x_const")
y = tf.constant(5, name="y_const")
sum = tf.add(x, y, name="x_y_sum")

# Task 1: Define a third scalar integer constant z.
z = tf.constant(4, name="z_const")
# Task 2: Add z to 'sum' to yield a new sum.
new_sum = tf.add(sum, z, name="x_y_z_sum")

# Now create a session.
# The session will run the default graph.
with tf.Session() as sess:
	# Task 3: Ensure the program yields the correct grand total.
	print(new_sum.eval())
```



