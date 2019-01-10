## 변수, 초기화, 할당

지금까지 수행한 모든 연산은 정적 값(tf.constant)에서 실행되었고; eval()을 호출하면 항상 같은 결과가 반환되었습니다. 텐서플로우에서는 변수 객체를 정의할 수 있으며, 변수 값은 변경할 수 있다. 변수를 만들 때 초기 값을 명시적으로 설정하거나 이니셜라이저를 사용할 수 있다. 

```python
from __future__ import print_function

import tensorflow as tf

g = tf.Graph()
with g.as_default():
	# Create a variable with the initial value 3.
	v = tf.Variable([3])
	
	# Create a variable of shape [1], with a random initial value.
	# sampled from a normal distribution with mean 1 and standard deviation 0.35.
	w = tf.Variable(tf.random_normal[1], mean=1.0, stddev=0.35)

with tf.Session() as sess:
	try:
		v.eval()
	except tf.errors.FailedPreconditionError as e:
		print("Caught expected error: ", e)

```

텐서 플로우의 한 가지 특징은 변수 초기화가 자동으로 실행되지 않는다. 

변수를 초기화하는 가장 쉬운 방법은 global_variables_initializer를 호출하는 것입니다. eval()과 거의 비슷한 Session.run()의 사용을 참고하세요.

```python
with g.as_default():
  with tf.Session() as sess:
    initialization = tf.global_variables_initializer()
    sess.run(initialization)
    # Now, variables can be accessed normally, and have values assigned to them.
    print(v.eval())
    print(w.eval())
```

초기화 된 변수는 같은 세션 내에서는 값을 유지합니다. 하지만 새 세션을 시작하면 다시 초기화해야 합니다. 

```python
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # These three prints will print the same value.
    print(w.eval())
    print(w.eval())
    print(w.eval())
```

변수 값을 변경하려면 할당 작업을 사용합니다. 할당 작업을 만들기만 하면 실행되는 것은 아닙니다. 초기화와 마찬가지로 할당 작업을 실행해야 변수 값이 업데이트 됩니다. 

```python
with g.as_default():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # This should print the variable's initial value.
    print(v.eval())

    assignment = tf.assign(v, [7])
    # The variable has not been changed yet!
    print(v.eval())

    # Execute the assignment op.
    sess.run(assignment)
    # Now the variable is updated.
    print(v.eval())
```

### 실습 #2 주사위 2개 10번 굴리기를 시뮬레이션 합니다. 

열 1 및 2 는 각각 주사위 1개를 1번 던졌을 때의 값입니다. 

열 3은 같은 줄의 열 1과 2의 합니다.



```python
from __future__ import print_function

import tensorflow as tf

with tf.Graph().as_default(), tf.Session() as sess:
  # Task 2: Simulate 10 throws of two dice. Store the results
  # in a 10x3 matrix.

  # We're going to place dice throws inside two separate
  # 10x1 matrices. We could have placed dice throws inside
  # a single 10x2 matrix, but adding different columns of
  # the same matrix is tricky. We also could have placed
  # dice throws inside two 1-D tensors (vectors); doing so
  # would require transposing the result.
  dice1 = tf.Variable(tf.random_uniform([10, 1],
                                        minval=1, maxval=7,
                                        dtype=tf.int32))
  dice2 = tf.Variable(tf.random_uniform([10, 1],
                                        minval=1, maxval=7,
                                        dtype=tf.int32))

  # We may add dice1 and dice2 since they share the same shape
  # and size.
  dice_sum = tf.add(dice1, dice2)

  # We've got three separate 10x1 matrices. To produce a single
  # 10x3 matrix, we'll concatenate them along dimension 1.
  resulting_matrix = tf.concat(
      values=[dice1, dice2, dice_sum], axis=1)

  # The variables haven't been initialized within the graph yet,
  # so let's remedy that.
  sess.run(tf.global_variables_initializer())

  print(resulting_matrix.eval())
```



