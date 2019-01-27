## 텐서 플로우 첫걸음

## 설정

첫번쨰 셀에서 필요한 라이브러리를 로드 하겠습니다. 

 ```python
from __future__ import print_fucntion

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{.1f}'.format


 ```

다음으로 데이터 세트를 로드 합니다. 

```python
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
```

확률적 경사하강법의 성능에 악영향을 줄 수 있는 의도치 않은 정렬 효과를 방지하고자 데이터를 무작위로 추출하였습니다. 또한 일반적으로 사용하는 학습률 범위에서 보다 쉽게 학습할 수 있도록 media_house_value를 천 단위로 조정하겠습니다. 

```python
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe
```

---

### 데이터 조사

데이터를 본격적으로 다루기 전에 데이터를 확인 하여야 합니다. 각 열에 대해 예의 개수, 평균, 표준편차, 최대값, 최소값, 다양한 분위 등 몇가지 유용한 통계를 간단히 요약하여 출력해봅니다.

```python
california_housig_dataframe.describe() # count, mean, std, min, 25%, 50%, 75%, max 등
```

---

### 첫 번쨰 모델 만들기

이 실습에서는 라벨 역할을 하는 median_house_value에 대한 예측을 시도합니다. 입력 특성으로는 total_rooms를 사용합니다. 

cf) 데이터는 지역 단위이므로 이 특성은 해당 지역의 전체 방 수 를 나타냅니다. 

모델을 학습시키려면 텐서플로우 Estimator API가 제공하는 LinearRegressor 인터페이스를 사용합니다. 이 API는 저수준 모델 작업을 알아서 처리하고 모델 학습, 평가, 추론을 수행하는 데 편리하게 사용되는 메소드를 노출합니다. 

#### 1단계 : 특성 정의 및 특성 열 구성

학습 데이터를 텐서플로우로 가져오려면 각 특성에 들어있는 데이터 유형을 지정해야 한다. 주로 2가지 데이터 유형이 있다.

- 범주형 데이터 : 텍스트로 이루어진 데이터. 이 실습의 주택 데이터 세트는 범주형 데이터를 포함하진 않지만 주택 양식, 부동산 광고문구 등의 예를 보게 될 수 도 있다. 
- 수치 데이터 : 정수 또는 부동 소수점 숫자이며, 숫자로 취급하려는 데이터. 이후에도 설명하겠지만, 우편번호 등의 수치데이터는 범주형으로 취급하는 경우도 있다. 

텐서 플로우에서 특성의 데이터 유형을 지정하려면 특성 열이라는 구조체를 사용한다. 특성 열은 특성 데이터에 대한 설명만 저장하며 특성 데이터 자체는 포함하지 않는다. 

우선 total_rooms라는 수치 입력 데이터 하나만 사용하면, 다음 코드에서 california_housing_dataframe에서 total_rooms 데이터를 추출하고 numeric_coumn으로 특성 열을 정의하여 데이터가 숫자임을 지정한다. 

```python
# Define the input feature: total_rooms
my_feature = california_housing_dataframe[['total_rooms']]

# Configure a numeric feature column for total_rooms
feature_column = [tf.feature_column.numeric_column("total_rooms")]
```

cf) total_rooms 데이터는 1차원 배열(각 지역의 전체 방 수로 이루어진 목록) 형태입니다. 이는 numeric_column의 기본 형태이므로 인수로 전달할 필요가 없습니다. 

---

#### 2단계 : 타겟 정의

```python
# Define the label
targets = california_housing_dataframe["median_house_value"]
```

---

#### 3단계 : LinearRegressor 구성

다음으로는 LinearRegressor를 사용하여 선형 회귀 모델을 구성합니다. 미니 배치 확률적 경사하강법(SGD)을 구현하는 GradientDescentOptimizer를 사용하여 이 모델을 학습시킬 것입니다. learning_rate 인수는 경사 단계의 크기를 조절 합니다. 

cf) 안전을 위해 옵티마이저 clip_gradient_by_norm을 통해 경사 제한을 적용합니다. 경사 제한은 학습 중에 경사가 너무 커져서 경사하강법이 실패하는 경우가 나타나지 않도록 제한합니다. 

```python
# Define the input feature : total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]

# Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# Define the label.
targets = california_housing_dataframe["median_house_value"]

# Use gradient descent as the optimizer for training the model.
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.00000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
	feature_columns = feature_columns,
	optimizer = my_optimizer
)
```

---

#### 4단계 입력함수 정의

캘리포니아 주택 데이터를 LinearRegressor로 가져오려면 텐서플로우 데이터 전처리 방법 및 모델 학습 중의 일괄처리, 셔플, 반복 방법을 알려주는 입력 함수를 정의해야 합니다. 

우선 pandas 특성 데이터를 NumPy 배열의 dict로 변환합니다. 그런 다음 텐서플로우의 Dataset API를 사용하여 이 데이터로부터 데이터 세트 개체를 생성하고 batch_size 크기의 배치로 나누어 지정한 세대 수(num_epochs)만큼 반복합니다. 

cf) 기본값인 num_epochs=None을 repeat()에 전달하면 입력 데이터가 무한정 반복됩니다. 

다음으로 shuffle을 True로 설정하면 학습 중에 데이터가 모델에 무작위로 전달되도록 데이터가 뒤섞입니다. buffer_size 인수는 shuffle에서 무작위로 추출할 데이터 세트의 크기를 지정합니다. 

마지막으로 입력함수에서 데이터 세트에 대한 반복자를 만들고 다음 데이터 배치를 LinearRegressor에 반환합니다. 

```python
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epoch=None):
"""Trains a linear regression model of one feature.
Args:
	features: pandas DataFrame of features
	targets: pandas DataFrame of targets
	batch_size: Size of batches to be passed to the model
	shuffle: True or False. Whether to shuffle the data.
	num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
Returns:
	Tuple of(features, labels) for next data batch
"""

# Convert pandas data into a dict of np arrays.
features = {key:np.array(value) for key, value in dict(features).items()}

# Construct a dataset, and configure batching/repeating.
ds = Dataset.from_tensor_slices((features, targets)) # warning: 2GB Limit
ds = ds.batch(batch_size).repeat(num_epochs)

# Shuffle the data, if specified.
if suffle:
	ds = ds.shuffle(buffer_size=10000)

# Return the next batch of data.
features, lables = ds.make_one_shot_iterator().getnext()
return features, labels
```

---

#### 5단계 : 모델 학습

이제 인풋함수로 부터 train()을 호출하여 모델을 학습시킬 수 있습니다. my_feature 및 target을 인수로 전달할 수 있도록 my_input_fn을 lambda에 래핑하겠습니다. 

```python
_ = linear_regressor.train(
	input_fn = lambda:my_input_fn(my_feature, targets)
	steps = 100
)
```

---

#### 6단계 : 모델 평가

모델 학습 중에 학습데이터에 얼마나 맞춰졌는지 확인하기 위해 학습 데이터로 예측을 실행합니다. 







