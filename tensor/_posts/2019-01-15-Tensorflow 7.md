## 텐서플로우 첫걸음 : 도구

다음 그림은 텐서플로우 도구함의 현재 계층구조를 보여 줍니다. 

![텐서플로우 도구함의 현재 계층구조 에스티메이터가 맨 위에 있습니다.](https://developers.google.com/machine-learning/crash-course/images/TFHierarchy.svg?hl=ko)

그림 1. 텐서플로우 도구함 계층 구조.

텐서플로우는 다음 두 요소로 구성됩니다. 

- 그래프 프로토콜 버퍼
- 분산된 그래프를 실행하는 런타임

이 두 구성요소는 자바 컴파일러 및 JVM과 유사합니다. JVM이 여러 하드웨어 플랫폼에서 구현되는 것과 마찬가지로 텐서 플로우도 여러  CPU와 GPU에서 구현됩니다. 

### tf.estimator API

이 단계에선 대부분의 실습이 tf.estimator를 사용합니다. 낮은 수준의 텐서 플로우를 사용해도 실습의 모든 작업을 실행 가능하지만 tf.estimator를 사용하면 코드 수가 크게 줄어듭니다. 

tf.estimator는 scikit-learn API와 호환됩니다. scikit-learn은 Python의 매우 인기있는 오픈소스 ML 라이브러리입니다. 

tf.estimator로 구현된 선형 회귀 프로그램의 형식은 대체로 다음과 같습니다. 

```python
import tensorflow as tf

# Set up a linear classifier.
classifier = tf.estimator.LinearClassifier()

# Train the model on some example data.
classifier.train(input_fn=train_input_fn, steps=2000)

# Use it to predict.
predictions = classfier.predict(input_fn=predict_input_fn)
```

---

## Pandas 간단 소개

Pandas 열 중심 데이터 분석 API입니다. 입력 데이터를 처리하고 분석하는 데 효과적인 도구이며, 여러 ML 프레임워크에서도 Pandas 데이터 구조를 입력으로 지원합니다. 

### 기본 개념

다음 행은 Pandas API를 가져와서 API 버전을 출력합니다. 

```java
from __future__ import print_function

import pandas as pd
pd._version_
```

Pandas의 기본 데이터 구조는 두 가지 클래스로 구현됩니다. 

- DataFrame은 행 및 이름 지정된 열이 포함된 관계형 데이터 테이블이라고 생각할 수 있습니다. 
- Series는 하나의 열입니다. DataFrame에는 하나 이상의 Series와 각 Series의 이름이 포함됩니다. 

데이터 프레임은 데이터 조작에 일반적으로 사용하는 추상화입니다. Spark 및 R에 유사한 구현이 존재합니다. 

Series를 만드는 한가지 방법은 Series 객체를 만드는 것입니다. 

```python
pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
```

DataFrame 객체는 string 열 이름과 매핑되는 'dict'를 각각의 Series에 전달하여 만들 수 있습니다. Series의 길이가 일치하지 않는 경우, 누락된 값은 특수 NA/NaN 값으로 채워집니다. 

```python
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

pd.DataFrame({'City name': citynames, 'Population':population})
```

하지만 대부분의 경우 전체 파일을 DataFrame으로 로드합니다. 다음 예는 캘리포니아 부동산 데이터가 있는 파일을 로드합니다. 

```python
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_hosing_train.csv", sep=",")
california_housing_dataframe.describe()
```

위의 예에서는 DataFrame.describe를 사용하여 DataFrame에 관한 흥미로운 통계를 보여줍니다. 또 다른 유용한 함수는 DataFrame.head로, DataFrame 레코드 중 처음 몇개 만 표시합니다.

```python
california_housing_dataframe.head()
```

Pandas의 또 다른 강력한 기능은 그래핑입니다. 예를 들어  DataFrame.hist를 사용하면 한 열에서 값의 분포를 빠르게 검토할 수 있습니다. 

```python
california_housing_dataframe.hist('housing_median_age')
```

### 데이터 액세스

익숙한 Python dict/list 작업을 사용하여 DataFrame 데이터에 액세스할 수 있습니다. 

```python
cities = pd.DataFrame({'City name':city_names, 'Population':population })
print(type(cities['City name']))
cities['City name']
```

```python
print(type(cities['City name'][1]))
cities['City name'][1]
```

```python
print(type(cities[0:2]))
citis[0:2]
```

### 데이터 조작

Python의 기본 산술 연산을 Series에 적용할 수도 있습니다. 예를 들면 다음과 같습니다. 

```python
population / 1000.
```

NumPy는 유명한 계산과학 툴킷입니다. Pandas Series는 대부분의 NumPy 함수에 인수로 사용할 수 있습니다. 

```python
import numpy as np
np.log(population) # 1000이 나눠진 값이 나온다.
```

더 복잡한 단일 열 변환에는 Series.apply를 사용할 수 있습니다.  Python map 함수처럼, Series.apply는 인수로 lambda 함수를 허용하며, 이는 각 값에 적용 됩니다. 

아래의 예에서는 인구가 백만 명을 초과하는지 나타내는 새 Series를 만듭니다. 

```python
population.apply(lambda val: val > 1000000)
```

DataFrames 수정 역시 간단합니다. 예를 들어 다음 코드는 기존 DataFrame에 두 개의 Series를 추가합니다. 

```java
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
```

### 실습 1

다음 두 명제 모두 True인 경우에만 True인 새 Bool 열을 추가하여 도시 테이블을 수정합니다. 

- 도시 이름은 'San'의 이름을 본따서 지었다.
- 도시 면적이 50제곱 킬로미터보다 넓다.

```python
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
```

### 색인

Series와 DataFrame 객체 모두 식별자 값을 각 Series 항목이나 DataFrame 해에 할당하는 index 속성을 정의합니다. 

기본적으로 생성 시 Pandas는 소스 데이터의 순서를 나타내는 index 값을 할당합니다. 생성된 이후 고정된다. 즉, 데이터의 순서가 재정렬 될때 변하지 않습니다. 

```python
city_names.index
cities.index
```

DataFrame.reindex를 호출하여 수동으로 행의 순서를 재정렬 합니다. 

```python
cities.reindex([2, 0, 1])
```

index 재생성은 DataFrame을 섞기 위한 좋은 방법이다. 아래의 예에서 배열처럼 된 색인을 NumPy의 random.permutation 함수에 전달하여 값을 섞는다. 이렇게 섞인 배열로 reindex를 호출하면 DataFrame 행도 같은 방식으로 섞인다.

```python
cities.reindex(np.random.permutation(cities.index))
```



