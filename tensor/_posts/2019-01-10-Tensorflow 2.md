## 텐서 만들기 및 조작

### 1. 벡터 덧셈

텐서에서 여러 일반적인 수학 연산을 할 수 있습니다. 다음 코드는 각기 정확히 6개 요소를 가지는 두벡터(1-D 텐서)를 만들고 조작합니다. 

```python
from __future__ import print_function

import tensorflow as tf

with tf.Graph().as_default():
# Create a six-element vector (1-D tensor).
primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

# Create another six-element vector. Each element in the vector will be
# initialized to 1. The first argument is the shape of the tensor (more
# on shapes below).
ones = tf.ones([6], dtype=tf.int32)

# Add the two vectors. The resulting tensor is a six-element vector.
just_beyond_primes = tf.add(primes, ones)

# Create a sesson to run the default graph.
with tf.Session() as sess:
	print(just_beyond_primes.eval())
```

----

### 2. 텐서 형태

형태는 텐서의 크기와 차원 수 를 결정하는 데 사용됩니다. 텐서 형태는 목록으로 표현하며, i번째 요소는 i 차원에서 크기를 나타냅니다. 그리고 이 목록의 길이는 텐서의 순위(예: 차원 수)를 나타냅니다.

```python
from __future__ import print_function

import tensorflow as tf

with tf.Graph.as_default():
	# A scalar (0-D tensor).
	scalar = tf.zeros([])
	
	# A vector with 3 elements.
	vector = tf.zeros([3])
	
	# A matrix with 2 rows and 3 columns.
	matrix = tf.zeros([2, 3])
	
	with tf.Sesson() as sess:
		print('scalar has shape', scalar.get_shape(), 'and value\n', scalar.eval())
		print('vector has shape', vector.get_shape(), 'and value\n', vector.eval())
		print('matrix has shape', scalar.get_shape(), 'and value\n', matrix.eval())
```

---

### 3. 텐서 형태 변경

텐서 덧셈과 행렬 곱셈에서 각각 피연사자에 제약조건을 부여하면 텐서플로우 프로그래머는 텐서의 형태를 변경해야 한다. tf.reshape 메소드를 사용하여 텐서의 형태를 변경할 수 있다. 예를 들어 8x2텐서를 2x8텐서나 4x4 텐서로 형태를 변경할 수 있다. 

```python
from __future__ import print_function

import tensorflow as tf

with tf.Graph.as_default():
	# Create an 8x2 matrix(2-D tensor).
	matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], 
					[9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.int32)
    # Reshape the 8x2 matrix into a 2x8 matrix.
    reshaped_2x8_matrix = tf.reshape(matrix, [2, 8])
    
    # Reshape the 8x2 matrix into a 4x4 matrix.
    reshaped_4x4_matrix = tf.reshape(matrix, [4, 4])
    
with tf.Session() as sess:
    print("Original matrix (8x2): ")
    print(matrix.eval())
    print("Reshaped matrix (2x8): ")
    print(reshaped_2x8_matrix.eval())
    print("Reshaped matrix (4x4): ")
    print(reshaped_4x4_matrix.eval())
```

또한 tf.reshape를 사용하여 텐서의 차원 수를 변경할 수도 있습니다. 예를 들어 8x2 텐서를 3-D 2x2x4 텐서나 1-D 16-요소 텐서로 변경할 수 있습니다.

```python
  # Reshape the 8x2 matrix into a 3-D 2x2x4 tensor.
  reshaped_2x2x4_tensor = tf.reshape(matrix, [2,2,4])
  
  # Reshape the 8x2 matrix into a 1-D 16-element tensor.
  one_dimensional_vector = tf.reshape(matrix, [16])
```



#### 실습 #1 : 두개의 텐서를 곱하기 위해 두 텐서의 형태를 변경합니다. 

다음 두 벡터는 행렬 곱셈과 호환되지 않습니다. 이 벡터를 행렬 곱셈에 호환될 수 있는 피연산자 형태를 변경하세요. 

```python
from __future__ import print_function

import tensorflow as tf

with tf.Graph().as_default(), tf.Session() as sess:
  # Task: Reshape two tensors in order to multiply them
  
  # Here are the original operands, which are incompatible
  # for matrix multiplication:
  a = tf.constant([5, 3, 2, 7, 1, 4])
  b = tf.constant([4, 6, 3])
  # We need to reshape at least one of these operands so that
  # the number of columns in the first operand equals the number
  # of rows in the second operand.

  # Reshape vector "a" into a 2-D 2x3 matrix:
  reshaped_a = tf.reshape(a, [2,3])

  # Reshape vector "b" into a 2-D 3x1 matrix:
  reshaped_b = tf.reshape(b, [3,1])

  # The number of columns in the first matrix now equals
  # the number of rows in the second matrix. Therefore, you
  # can matrix mutiply the two operands.
  c = tf.matmul(reshaped_a, reshaped_b)
  print(c.eval())
```

----



