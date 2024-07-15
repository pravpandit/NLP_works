# Chapter 3 Gradient Descent and Stochastic Gradient Descent
&emsp;&emsp;The training of linear models and neural networks can usually be described as an optimization problem. That is, let $\omega^{(1)},\omega^{(2)},\cdots\omega^{(l)}$ be the optimization variables (they can be vectors, matrices, tensors). We usually encounter solving such an optimization problem:
$$
\min_{w^{(1)},\cdots ,w^{(l)}}\quad L(w^{(1)},\cdots ,w^{(l)})
$$
&emsp;&emsp;For such a relatively simple unconstrained optimization problem, we often use the Gradient Descent algorithm (GD) and the Stochastic Gradient Descent algorithm (SGD) to find the optimal solution.

## 3.1 Gradient

&emsp;&emsp;We often use the first-order information of a function, namely the gradient, to find the optimal value of the function. The gradient of the above problem can be written as
$$
\underbrace{\nabla_{\boldsymbol{w}^{(i)}} L\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(l)}\right) \triangleq \frac{\partial L\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(l)}\right)}{\partial \boldsymbol{w}^{(i)}}}_{\text {Both symbols represent the gradient of } L \text { with respect to } \boldsymbol{w}^{(l)} \text { }}, \quad \forall i=1, \cdots, l .
$$
Note that the shape of the gradient $\displaystyle\nabla _{w^{(i)}}L$ should be exactly the same as the shape of $w^{(i)}$.

&emsp;&emsp;If you use deep learning platforms such as `TensorFlow` and `PyTorch`, you don't need to worry about how the gradient is calculated. As long as the defined function is differentiable with respect to a variable, `TensorFlow` and `PyTorch` can automatically calculate the gradient of the function with respect to the variable. However, we should be careful to check whether the shape of the gradient is the same as the shape of the variable before writing the program.

## 3.2 Gradient Descent

&emsp;&emsp;We usually stipulate that the gradient is the fastest functionAscending direction. Therefore, if you want to minimize a function, it is natural to think of searching in the opposite direction of the gradient. Searching in the opposite direction of the gradient is called gradient descent (GD).
$$
x_{k+1}=x_{k}+\alpha_{k}*\left(-\nabla f\left(x_{k}\right)\right).
$$
&emsp;&emsp;We can also use the example of blind climbing to understand the meaning of the gradient descent algorithm. Blind climbing can be regarded as finding the maximum value of a function. The blind can obtain the current slope (i.e. gradient) at each step, but does not know any situation of other points. The gradient descent method is equivalent to climbing forward (or going down the mountain) along the steepest direction of the slope in climbing.

&emsp;&emsp;Then, for the optimization problem raised above, we can write the blind gradient descent algorithm process:
$$
w_{\text {new }}^{(i)} \leftarrow w_{\text {now }}^{(i)}-\alpha \cdot \nabla_{w^{(i)}} L\left(w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}\right), \quad \forall i=1, \cdots, l
$$
Where, $\displaystyle w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}$ are the variables that need to be optimized. 

&emsp;&emsp;The $\alpha$ in the above formula is usually called the step size or learning rate. Its setting affects the convergence rate of the gradient descent algorithm, which ultimately affects the test accuracy of the neural network, so $\alpha$ needs to be carefully adjusted. In mathematics, $\alpha$ is often found by line search. You can refer to Jorge Nocedel's "Numerical Optimization" [1], which will not be repeated here. 

&emsp;&emsp;When the optimization function is a convex L-Lipschitz continuous function, the gradient descent method can guarantee convergence, and the convergence rate is $\displaystyle O(\frac{1}{k})$, where $k$ is the number of iterations. 

> **Note: **The definition of Lipschitz continuity is that if the function $f$ is continuous with a constant L-Lipschitz on the interval $Q$, then for $x, y \in Q$, we have
> $$
> \|f(x)-f(y)\| \leq L\|x-y\|
> $$

&emsp;&emsp;Give a simple python program to review the gradientLowering method.
```python
"""
Gradient descent example for one-dimensional problem
"""

def func_1d(x):
"""
Objective function
:param x: independent variable, scalar
:return: dependent variable, scalar
"""
return x ** 2 + 1

def grad_1d(x):
"""
Gradient of objective function
:param x: independent variable, scalar
:return: dependent variable, scalar
"""
return x * 2

def gradient_descent_1d(grad, cur_x=0.1, learning_rate=0.01, precision=0.0001, max_iters=10000):
"""
Gradient descent for one-dimensional problem
:param grad: Gradient of objective function
:param cur_x: current x value, initial value can be provided by parameter
:param learning_rate: learning rate, also equivalent to the set step size
:param precision: Set the convergence precision:param max_iters: maximum number of iterations
:return: local minimum x*
"""
for i in range(max_iters):
grad_cur = grad(cur_x)
if abs(grad_cur) < precision:
break # When the gradient approaches 0, it is considered converged
cur_x = cur_x - grad_cur * learning_rate
print("Iteration", i, ": x value is ", cur_x)
print("Local minimum x =", cur_x)
return cur_x

if __name__ == '__main__':
gradient_descent_1d(grad_1d, cur_x=10, learning_rate=0.2, precision=0.000001, max_iters=10000)
```

## 3.3 Stochastic Gradient Descent

&emsp;&emsp;When you need to optimize large-scaleWhen it comes to the problem of minimizing the gradient, calculating the gradient has become a very troublesome thing. Is it possible to use some examples in the gradient sample to approximate all the gradient samples? The answer is yes! 

&emsp;&emsp;If the objective function can be written in the form of continuous addition or expectation, then the stochastic gradient descent can be used to solve the minimization problem.

&emsp;&emsp;Assume that the objective function can be written as $n$ consecutive addition form:
$$
L\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(l)}\right)=\frac{1}{n} \sum_{j=1}^{n} F_{j}\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(l)}\right)
$$
&emsp;&emsp;Wherein, the function $F_j$ implies the $j$th training sample $(x_j , y_j)$. Each time, an integer is randomly drawn from the set ${1, 2, \cdots , n}$, denoted as $j$. Suppose the current optimization variable is $w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}$ Calculate the stochastic gradient here and perform stochastic gradient descent on it:$$
\mid w_{\text {new }}^{(i)} \leftarrow w_{\text {now }}^{(i)}-\alpha \cdot \underbrace{\nabla_{w^{(i)}} F_{j}\left(w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}\right)}_{\text {stochastic gradient }}, \quad \forall i=1, \cdots, l .
$$
&emsp;&emsp;In fact, in actual operation, we will find that when using the GD algorithm to solve some non-convex optimization problems, the program often stops at the saddle point and cannot converge to the local optimal point, which will lead to very low test accuracy; and using the SGD method can help us jump out of the saddle point and continue to move towards a better optimal point.

&emsp;&emsp;It is gratifying that SGD can also guarantee convergence. The specific proof process is relatively complicated. If you are interested, you can read the literature [4]. Here we only give a **sufficient condition** for SGD convergence:
$$
\sum_{k=1}^{\infty}\alpha_k=\infty,\sum_{k=1}^{\infty}\alpha_k^2<\infty
$$

&emsp;&emsp;Finally, a simple Python program is given to review the stochastic gradient descent method.
```python
import numpy as np
import math

# Generate test data
x = 2 * np.random.rand(100, 1) # Randomly generate a 100*1 two-dimensional array with values ​​between 0 and 2

y = 4 + 3 * x + np.random.randn(100, 1) # Randomly generate a 100*1 two-dimensional array with values ​​between 4 and 11

x_b = np.c_[np.ones((100, 1)), x]
print("The content of the x matrix is ​​as follows:\n{}".format(x_b[0:3]))
n_epochs = 100
t0, t1 = 1, 10

m = n_epochs
def learning_schedule(t): # Simulate the dynamic modification of the step size
return t0 / (t + t1)

theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
for i in range(m):random_index = np.random.randint(m)
x_i = x_b[random_index:random_index+1]
y_i = y[random_index:random_index+1]
gradients = 2 * x_i.T.dot(x_i.dot(theta)-y_i) # Call formula
learning_rate = learning_schedule(epoch * m + i)
theta = theta - learning_rate * gradients

if epoch % 30 == 0:
print("Sampling view: \n{}".format(theta))

print("Final result: \n{}".format(theta))

# Calculate error
error = math.sqrt(math.pow((theta[0][0] - 4), 2) + math.pow((theta[1][0] - 3), 2))
print("Error:\n{}".format(error))
```

## References

[1] Wang Shusen, Li Yujun, Zhang Zhihua, Deep Reinforcement Learning, https://github.com/wangshusen/DRL/blob/master/Notes_CN/DRL.pdf, 2021
[2] Nocedal, Jorge & Wright, Stephen. (2006). Numerical Optimization. 10.1007/978-0-387-40065-5.
[3] Jorge Nocedal§. Optimization Methods for Large-Scale Machine Learning[J]. Siam Review, 2016, 60(2).
[4] Nemirovski A S , Juditsky A , Lan G , et al. Robust Stochastic Approximation Approach to Stochastic Programming[J]. SIAM Journal on Optimization, 2009, 19(4):1574-1609.