# chapter2 Line Search
&emsp;&emsp;Optimization problems can be divided into two categories: unconstrained optimization problems and constrained optimization problems. Unconstrained optimization problems are problems that seek the extreme value of a function, namely
$$
\min f(x) \tag{1}
$$
where $x \in \mathbb{R}^n$ is the decision variable and $f(x)\in \mathbb{R}$ is the objective function. The solution to problem $(1)$ is called the optimal solution, denoted as $x^{*}$, and the function value $f(x^{*})$ at this point is called the optimal value. The optimal solution to problem $(1)$ is divided into global optimal solutions and local optimal solutions. The line search algorithm introduced in this section is an algorithm for seeking local optimal solutions. This section mainly refers to the contents of relevant chapters in the two books "Optimization: Modeling, Algorithms and Theory" and "Numerical Optimization Methods".
## 2.1 Line search algorithm structure
&emsp;&emsp;The basic method for solving optimization problems is the iterative algorithm, that is, the method of using a step-by-step approximation calculation method to approximate the exact solution of the problem. Taking the minimization problem as an example, in an algorithm, an initial iteration point $x_{0}\in \mathbb{R}^n$ can be selected first, and at this iteration point, a direction in which the function value decreases is determined, and then the step length in this direction is determined to obtain the next iteration point, and so on, to generate an iteration point sequence {$x_{k}$}, {$x_{k}$} or its subsequence should converge to the problemThe optimal solution. When a given termination criterion is met, it either indicates that $x_{k}$ has met the required accuracy of the approximate optimal solution, or that the algorithm is no longer able to further improve the iteration point, and the iteration ends.
The basic structure of the line search algorithm is as follows:
$$
\begin{array}{l}
(1) Given the initial point x_{0} \in \mathbb{R^{n}}, k:=0 \\
(2) If the termination criterion is met at point x_{k}, then output the relevant information and stop the iteration \\
(3) Determine the descent direction d_{k} of f(x) at point x_{k} \\
(4) Calculate the step size \alpha_{k} so that f(x_{k}+\alpha_{k} d_{k}) is less than f(x_{k}) \\
(5) Let x_{k+1}:=x_{k}+\alpha_{k} d_{k} , k:=k+1, go to (2) \\
\end{array}
$$
It contains two basic elements: one is the descent direction; the other is the step size. Different descent directions and step sizes can constitute different algorithms. The optimization method with the above structure is called the line search method.
## 2.2 Termination criteria
&emsp;&emsp;Because the local minimum point $x^{*}$ is a stable point (assuming the first-order derivative of the objective function $f(x)$ exists), $\lVert \nabla{f(x_{k})} \rVert \leqslant \epsilon$ is used as the termination criterion. The size of $\epsilon$ determines the accuracy of the obtained iteration point $x_{k}$ in approximating $x^{*}$. However, this criterion also has certain limitations. For functions that are relatively steep in the minimum point area, even if the points in the area are quite close to the minimum point, their gradient values ​​may still be large, making it difficult to stop the iteration.

&emsp;&emsp;Other termination criteria include $\lVert x_{k}-x_{k+1}\rVert \leqslant \epsilon$ or $f_{k}-f_{k+1}\leqslant\epsilon$, but the satisfaction of these criteria only means that the iterations performed by the algorithm at this time have made very little improvement to the iteration point or the objective function value at the iteration point, and cannot guarantee that $\lVert x_{k}-x_{k+1}\rVert$ or $f_{k}-f_{k+1}$ must be small enough.

## 2.3 Search direction
&emsp;&emsp;At the iteration point $x_{k}$, if there exists $\overline{\alpha}_{k}$, so that
$$
f\left(x_{k}+\alpha_{k} d\right)<f\left(x_{k}\right), \forall\alpha \in (0, \overline{\alpha}_{k}),
$$
Then $d$ is the descending direction of $f(x)$ at $x_{k}$.

&emsp;&emsp;According to the Taylor expansion of $f\left(x_{k}+\alpha_{k} d\right)$ at $x_{k}$,
$$
f\left(x_{k}+\alpha d\right)=f\left(x_{k}\right)+\alpha \nabla f\left(x_{k}\right)^{T}d +O(\lVert \alpha d \rVert^{2})
$$
We know that the descending direction $d$ is the direction that satisfies $\nabla f\left(x_{k}\right)^{T}d < 0$.

&emsp;&emsp;Line search algorithms can be divided into gradient algorithms, sub-gradient algorithms, Newton algorithms, quasi-Newton algorithms, etc. according to the different search directions.
## 2.4 Search Step Size
&emsp;&emsp;In online search methods, there are many choices of search directions, but the step size selection method is very similar in different algorithms. Let $\phi(\alpha)=f(x_{k}+\alpha d_{k})$, starting from the current iteration point $x_{k}$, along the search direction $d_{k}$, it is necessary to determine the appropriate step size $\alpha_{k}$ so that $f(x_{k}+\alpha_{k} d_{k})<f(x_{k})$, that is, $\phi(\alpha_{k})<\phi(0)$. The choice of search step size usually requires a balance between the amount of objective function descent and the amount of computation required to determine $\alpha_{k}$.

&emsp;&emsp;A natural idea is to take $\alpha_{k}$ to minimize the objective function $f(x_{k})$ along the direction $d_{k}$, that is,
$$
\phi( \alpha_{k} )=\operatorname* {min}\limits_{\alpha> 0} \phi( \alpha) ,
$$
This method is called exact line search. Since exact line search usually requires a lot of computation in actual calculations, and it is very difficult to implement exact line search for general problems, it is rarely used in practical applications.

&emsp;&emsp;Another idea is not to find the minimum point of $
\phi( \alpha)$, but to select $\alpha_{k}$ so that the objective function can obtain an acceptable decrease of $f(x_{k})-f(x_{k}+\alpha_{k} d_{k})$. This line search method is called inexact line search. Inexact line search is more popular because it requires relatively less calculation.
### 2.4.1 Inexact line searchCriteria
&emsp;&emsp;In the inexact line search algorithm, only $f(x_{k}+\alpha_{k} d_{k})<f(x_{k})$ is not enough to ensure that the generated iterative sequence $\{x_{k}\}$ converges to the optimal solution. The selection of $\alpha_{k}$ needs to meet certain requirements, which are called line search criteria. The suitability of the line search criterion directly determines the convergence of the algorithm. If an inappropriate line search criterion is selected, the algorithm will not converge to the minimum point. For example, consider the one-dimensional unconstrained optimization problem
$$
\operatorname* {m i n}_{x} f(x)=x^{2}
$$
The initial iteration point $x_{0}=1$. Since the problem is one-dimensional, there are only two descent directions: {-1, +1}. We select $d_{k}=-sign(x_{k})$, and only require the selected step size to satisfy the monotonically decreasing function value at the iteration point, that is, 
$
f(x_{k}+\alpha d_{k})<f(x_{k})
$
Consider the following two step sizes:

$$
\alpha_{k, 1}=\frac{1} {3^{k+1}}, \quad\alpha_{k, 2}=1+\frac{2} {3^{k+1}}, 
$$
Through calculation, we can get

$$
x_{k}^{1}=\frac{1} {2} \left( 1+\frac{1} {3^{k}} \right), \quad x_{k}^{2}=\frac{(-1 )^{k}} {2} \left( 1+\frac{1} {3^{k}} \right). 
$$

Obviously, the sequences $\{f(x_{k}^{1})\}$ and $\{f(x_{k}^{2})\}$ both decrease monotonically, but the point where the sequence $\{x_{k}^{1}\}$ converges is not the minimum point, and the sequence $\{x_{k}^{2}\}$ oscillates around the origin and has no limit, as shown in the figure below.
The reason for the above situation is that the decrease of the function value $f(x)$ during the iteration is not sufficient so that the algorithm cannot converge to the minimum point. In order to avoid this situation, some more reasonable line search criteria must be introduced to ensure the convergence of the iteration.

![image](./images/ch2/fxconvergence graph.png "image")

#### 1. Armijo criterion
&emsp;&emsp;Armijo criterion:

$$
f ( x_{k}+\alpha d_{k} ) \leqslant f ( x_{k} )+\rho \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k}, \rho\in (0,1)
$$
is a commonly used line search criterion. The purpose of introducing the Armijo criterion is to ensure that each iteration is sufficiently reduced.
The Armijo criterion has a very intuitive geometric meaning. It means that the point $(\alpha,\phi(\alpha))$ must be below the line
$$l(\alpha)=f ( x_{k} )+\rho\alpha \nabla f(x_{k})^{\mathsf{T}}d_{k}$$
. As shown in the figure below ($g=\nabla f (x)$), the points in the intervals (0,$\beta_{4}$] and [$\beta_{5}$,$\beta_{6}$] all satisfy the Armijo criterion. Because $\nabla f ( x_{k} )^{\mathsf{T}} d_{k}<0$, the slope of $l(\alpha)$ is negative, and selecting $\alpha$ that meets the Armiio criterion will indeed cause the function value to decrease. In practical applications, the parameter $c_{1}$ is usually selected as a very small positive number, such as $c_{1}=10^{-3}$, which makes the Armijo criterion very easy to satisfy. However, using only the Armijo criterion cannot guarantee the convergence of the iteration, because the feasible region contains the area where the step size $\alpha$ is close to 0. When $\alpha$ takes the valueIf it is too small, the decrease in the objective function value may be too small, resulting in the limit value of the sequence $\{f(x_{k})\}$ not being the minimum value. In order to avoid $\alpha$ being too small, the Armijo criterion needs to be used in conjunction with other criteria.

![Armijo criterion](./images/ch2/Armijo criterion_1.png)

#### 2. Goldstein criterion
&emsp;&emsp;In order to overcome the defects of the Armijo criterion, other criteria need to be introduced to ensure that $\alpha$ at each step is not too small. Since the Armijo criterion only requires that the point ($\alpha,\phi(\alpha)$) must be below a certain straight line, the same form can also be used to make the point must be above another straight line. This is the Armijo-Goldstein criterion, or Goldstein criterion for short:

$$
\begin{aligned} {{f ( x_{k}+\alpha d_{k} )}} & {{} {{} \leqslant f ( x_{k} )+\rho \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k},}} \\ {{f ( x_{k}+\alpha d_{k} )}} & {{} {{} \geqslant f ( x_{k} )+( 1-\rho ) \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k},}} \\ \end{aligned} 
$$
where $\rho\in (0,1/2)$.
Similarly, the Goldstein criterion has a very intuitive geometric meaning, which means that a point ($\alpha,\phi(\alpha)$) must be between two lines

$$
\begin{aligned} {{l_{1} ( \alpha)}} & {{} {{} {{}=f ( x_{k} )+\rho \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k},}}} \\ {{l_{2} ( \alpha)}} & {{} {{} {{}=f ( x_{k} )+( 1-\rho ) \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k}}}} \\ \end{aligned} 
$$
between. As shown in the figure below, the points in the intervals [$\beta_{3}$,$\beta_{4}$] and [$\beta_{5}$,$\beta_{6}$] all satisfy Goldsteimn criterion, and it can be noted that the Goldstein criterion does remove the $\alpha$ that is too small.
![Goldstein criterion](./images/ch2/Goldstein criterion_1.png "Goldstein criterion")

#### 3.Wolfe criterion
&emsp;&emsp;Goldstein criterion can make the function value drop sufficiently, but it may avoid the area where $\phi(\alpha)$ takes the minimum value. For this purpose, the Armijo-Wolfe criterion is introduced, which is referred to as the Wolfe criterion: 
$$
f ( x_{k}+\alpha d_{k} ) \leqslant f ( x_{k} )+\rho \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k}, 
$$
$$
\nabla f ( x_{k}+\alpha d_{k} )^{\mathrm{T}} d_{k} \geqslant \sigma \nabla f ( x_{k} )^{\mathrm{T}} d_{k}, 
$$
where $1>\sigma>\rho>0$ is a given constant. In the Wolfe criterion, the first inequality is the Armijo criterion, and the second inequality isThe essential requirement of the Wolfe criterion. Noting that $\nabla f ( x_{k}+\alpha d_{k} )^{\mathrm{T}} d_{k}$ is exactly the derivative of $\phi(\alpha)$, the Wolfe criterion actually requires that the slope of the tangent line of $\phi(\alpha)$ at point $\alpha$ cannot be less than $\sigma$ times the slope of $\phi(\alpha)$ at zero. As shown in the figure below, the points in the intervals [$\beta_{7}$,$\beta_{4}$], [$\beta_{8}$,$\beta_{9}$], and [$\beta_{10}$,$\beta_{6}$] all satisfy the Wolfe criterion. Note that at the minimum point $\alpha^{*}$ of $\phi(\alpha)$, $\phi'(\alpha^{*})=\nabla f ( x_{k}+\alpha^{*} d_{k} )^{\mathrm{T}} d_{k}=0$, so $\alpha^{*}$ always satisfies the second inequality. Choosing a smaller $\rho$ can make $\alpha^{*}$ satisfy the first inequality at the same time, that is, the Wolfe criterion will contain the exact solution of the line search subproblem in most cases. In practical applications, the parameter $\sigma$ is usually taken as0.9.

![Wolfe Criterion](./images/ch2/Wolfe Criterion_1.png "Wolfe Criterion")

&emsp;&emsp;In the Wolfe Criterion, even if $\sigma$ is set to 0, there is no guarantee that the points that meet the criterion are close to the results of the exact line search. However, if the following strong Wolfe criterion is adopted, the smaller $\sigma$ is, the closer $\alpha$ that satisfies the criterion is to the result of the exact line search,
$$
f(x_{k}+\alpha d_{k})\leqslant f(x_{k})+\rho \alpha\nabla f ( x_{k} )^{\mathrm{T}}d_{k}\\|\nabla f(x_{k}+\alpha d_{k})^{\mathrm{T}}d_{k}|\leqslant-\sigma \nabla f( x_{k} )^{\mathrm{T}}d_{k}
$$
where $1>\sigma>\rho>0$.
### 2.4.2 Convergence
&emsp;&emsp;
This section will take the Wolfe criterion as an example and introduce the Zoutendijk theorem to illustrate the convergence of the inexact line search algorithm.

$\textbf{Zoutendijk theorem:}$ Consider the general iterative format $x_{k+1}=x_{k}+\alpha_{K}d_{k}$, where $d_{k}$ is the search direction, $\alpha_{k}$ is the step size, and the Wolfe criterion is satisfied during the iteration process. Assume that the objective function $f$ is bounded, continuously differentiable and has a gradient $L$-Lipschitz continuous, that is,
$$
\| \nabla f ( x )-\nabla f ( y ) \| \leqslant L \| x-y \|, \quad\forall\, x, y \in\mathbb{R}^{n}, 
$$
Then
$$
\sum_{k=0}^{\infty} \operatorname{cos}^{2} \theta_{k} \| \nabla f ( x^{k} ) \|^{2} <+\infty, \tag{2}
$$
Where $\operatorname{cos}_{\theta_{k}}$ is the cosine of the angle between the negative gradient $-\nabla f(x_{k})$ and the descending direction $d_{k}$, that is,
$$
\operatorname{cos} \theta_{k}=\frac{-\nabla f ( x_{k} )^{\mathrm{T}} d_{k}} {\| \nabla f ( x_{k}) \| \| d_{k} \|}. 
$$
Inequality (2) is also called the $\textbf{Zoutendijk condition}$.

&emsp;&emsp;$\textbf{Proof:}$

&emsp;&emsp;By Wolfe condition
$$
\nabla f ( x_{k}+\alpha d_{k} )^{\mathrm{T}} d_{k} \geqslant \sigma \nabla f ( x_{k} )^{\mathrm{T}} d_{k}, 
$$
We can get
$$
\Big( \nabla f ( x_{k+1} )-\nabla f ( x_{k} ) \Big)^{\mathsf{T}} d_{k} \geqslant( \sigma-1 ) \nabla f ( x_{k} )^{\mathsf{T}} d^{k}.
$$
By Cauchy inequality and gradient L-Lipschitz continuity property

$$
\left( \nabla f ( x_{k+1} )-\nabla f ( x_{k} ) \right)^{\mathrm{T}} \! d^{k} \leqslant\| \nabla f ( x_{k+1} )-\nabla f ( x_{k} ) \| \| d_{k} \| \leqslant\alpha_{k} L \| d_{k} \|^{2}. 
$$
Combining the above two equations, we can get

$$
\alpha_{k} \geqslant{\frac{\sigma-1} {L}} {\frac{\nabla f ( x_{k} )^{\mathrm{T}} d_{k}} {\| d_{k} \|^{2}}}. 
$$
Noting that $\nabla f ( x_{k} )^{\mathrm{T}} d_{k}<0$, substitute the above equation into the first inequality $f ( x_{k}+\alpha d_{k} ) \leqslant f ( x_{k} )+\rho \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k}$ condition, then

$$
f ( x_{k+1} ) \leqslant f ( x_{k} )+\rho {\frac{\sigma-1} {L}} {\frac{\left( \nabla f ( x_{k} )^{\mathsf{T}} d_{k} \right)^{2}}{\| d_{k} \|^{2}}}. 
$$
According to the definition of $\theta_{k}$, this inequality can be equivalently expressed as

$$
f ( x_{k+1} ) \leqslant f ( x_{k} )+\rho \frac{\sigma-1} {L} \operatorname{c o s}^{2} \theta_{k} \| \nabla f ( x_{k} ) \|^{2}. 
$$
Summing over k, we have
$$
f ( x_{k+1} ) \leqslant f ( x_{0} )-\rho \frac{1-\sigma} {L} \sum_{j=0}^{k} \operatorname{c o s}^{2} \theta_{j} \| \nabla f ( x_{j} ) \|^{2}. 
$$
Because the function $f$ is bounded below, and $0<\rho<\sigma<1$ shows that $\rho(1-\sigma)>0$, when $k\rightarrow +\infty$

$$
\sum_{j=0}^{\infty} \operatorname{cos}^{2} \theta_{j} \| \nabla f ( x_{j} ) \|^{2} <+\infty. 
$$

&emsp;&emsp;Zoutendik theorem points out that as long as the iteration point satisfies the Wolfe criterion, the gradient Lipschitz continuous and lower bounded function can always be deduced that equation (2) is valid. In fact, similar conditions can also be deduced using the Goldstein criterion. Zoutendik theorem characterizes the properties of the line search criterion, and the most basic convergence of the line search algorithm can be obtained by combining the selection method of the descending direction $d_{k}$.

$\textbf{Convergence of line search algorithm:}$

&emsp;&emsp;For the line search algorithm, let $\theta_{k}$ be the angle between the negative gradient $-\nabla f(x_{k})$ at each step and the descending direction $d_{k}$, and assume that for any k, there exists a constant $\gamma>0$, such that
$$
\theta_{k} < \frac{\pi} {2}-\gamma, 
$$
Then under the condition that Zoutendik's theorem holds, there is
$$
\operatorname* {l i m}_{k \to\infty} \nabla f ( x^{k} )=0. 
$$
&emsp;&emsp;$\textbf{Proof:}$

&emsp;&emsp;Assume that the conclusion does not hold, that is, there exists a subsequence $\{k_{l}\}$ and a positive constant $\delta>0$, so that

$$
\| \nabla f ( x_{k_{l}} ) \| \geqslant\delta, \quad l=1, 2, \cdots. 
$$
According to the assumption of $\theta_{k}$, for any k,

$$
\operatorname{c o s} \theta_{k} > \operatorname{s i n} \gamma> 0. 
$$
We only consider the $k_{l}$th term in (2), and we have

$$
\begin{aligned} {{\sum_{k=0}^{\infty} \operatorname{c o s}^{2} \theta_{k} \| \nabla f ( x_{k} ) \|^{2}}} & {{} \geqslant\sum_{l=1}^{\infty} \operatorname{cos}^{2} \theta_{k_{l}} \| \nabla f ( x_{k_{l}} ) \|^{2}} \\ {} & {{} \geqslant\sum_{l=1}^{\infty} ( \operatorname{sin}^{2} \gamma) \cdot\delta^{2} \to+\infty,} \\ \end{aligned} 
$$
This obviously contradicts the Zoutendijk theorem. Therefore, it must be

$$
\operatorname* {l i m}_{k \to\infty} \nabla f ( x^{k} )=0. 
$$
This proof is based on the Zoutendik condition, which essentially requires that the descent direction $d_{k}$ of each step and the negative gradient direction cannot be orthogonal. The geometric intuition of this condition is obvious: when the descent direction $d_{k}$ is orthogonal to the gradient, according to the first-order approximation of Taylor expansion, the objective function value $f(x)$ hardly changes, so it is required that the angle between $d_{k}$ and the orthogonal direction of the gradient has a consistent lower bound. The convergence speed of the line search algorithm depends greatly on the selection of $d$ and the properties of the objective function. For details, please refer to the relevant content in the book "Numerical Optimization".
### 2.4.3 Search Algorithm
&emsp;&emsp;This section mainly introduces how to find the step size that satisfies the line search criteria.
#### 1 Backoff Algorithm
&emsp;&emsp;The backoff method is one of the most commonly used line search algorithms. It is to find a point that satisfies the Armijo criterion by continuously reducing the trial step size exponentially.For example:
>1. Select the initial step size
$\overline{\alpha}$, parameter $\gamma,c\in(0,1)$. Initialize $\alpha\leftarrow\overline{\alpha}$.
>
>2. while$f(x_{k} + \alpha d_{k}) > f(x_{k}) + c\alpha \nabla f(x_{k})^{\mathbb{T}}d_{k}$ do
>
>3. &emsp;&emsp;Let $\alpha \leftarrow \gamma \alpha$
>
>4. end while
>5. Output $\alpha_{k}=\alpha$

&emsp;&emsp;The trial value of $\alpha$ in the backoff method is from large to small, so it can ensure that the output $\alpha$ can be as large as possible. At the same time, because $d$ is in a descending direction, when $\alpha$ is sufficiently small, the Armijo criterion is always valid, and the backoff method will not be limited. In practical applications, we usually set a lower bound for $\alpha$ to prevent the step size from being too small. However, the disadvantages of the backoff method are also obvious: first, it cannot guarantee to find a step size that satisfies the Wolfe criterion, but for some optimization algorithms, finding a step size that satisfies the Wolfe criterion is not guaranteed.Then the step size is very necessary; second, the backoff method reduces the step size exponentially, so it is sensitive to the selection of the initial value $\overline{\alpha}$ and the parameter $\gamma$. When $\gamma$ is too large, the change in each trial step size is very small, and the backoff method is inefficient at this time. When $\gamma$ is too small, the backoff method is too aggressive, resulting in the final step size being too small, and the opportunity to select a large step size is missed.
#### 2 Polynomial interpolation method
&emsp;&emsp;In order to improve the efficiency of the backoff method, there is a line search algorithm based on polynomial interpolation. Its principle is to use the existing function information to construct a polynomial function approximating $\phi(\alpha)$, find the minimum point of the polynomial function and check whether it meets the inexact line search criterion; if not, construct a new polynomial function based on the new function information; and repeat this process until the inexact line search criterion is met.

&emsp;&emsp;Take the quadratic interpolation method and Armijo criterion as an example. Assuming that the initial step size $\alpha_{0}$ is given, if it is verified that $\alpha_{0}$ does not meet the Armijo criterion, the next step is to reduce the trial step size and construct a quadratic interpolation function $p_{2}(\alpha)$ based on the three pieces of information $\phi(0),\phi'(0),\phi(\alpha_{0})$ to satisfy
$$
p_2(0)=\phi(0),\quad p_2'(0)=\phi'(0),\quad p_2({\alpha}_0)=\phi({\alpha}_0).
$$
Since quadratic functions have only three parameters, the above three conditions can uniquely determine $p_{2}(\alpha)$, and it is not difficult to verify that the minimum point of $p_{2}(\alpha)$ is exactly within (0,$\alpha_{0}$); at this time, take the minimum point of $p_{2}(\alpha)$, $\alpha_{1}$, as the next trial point, and use the same method to continuously recurse until a point that satisfies the Armijo criterion is found.
## References

【1】Liu Haoyang, Hu Jiang, Li Yongfeng, Wen Zaiwen. Optimization: Modeling, Algorithms and Theory[M]. Higher Education Press, 2020. <br>
【2】Gao Li. Numerical Optimization Methods[M]. Peking University Press, 2014. <br>
【3】Nocedal, Jorge & Wright, Stephen. Numerical Optimization. 2006. <br>
【4】Yuan Yaxiang, Sun Wenyu. Optimization Theory and Methods[M]. Science Press, 1997.