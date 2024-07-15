This chapter mainly studies the problem of constrained minimization:
$$
\begin{split}
&\min\limits_{x\in\mathbb{R}} f(x)\\\
&s.t. \begin{cases}
c_i(x) = 0, i\in \mathcal{E},\\
c_i(x) \geq 0, i\in \mathcal{I},
\end{cases}
\end{split}
$$

It is agreed that $c_i$ is a real-valued function defined on $\mathbb{R}$ or its subset, $\mathcal{E}$ and $\mathcal{I}$ are the index sets of equality constraints and inequality constraints respectively. So the feasible domain of the problem is
$$
\mathcal{X} = \{x\in \mathbb{R}\mid c_i(x)=0, i\in \mathcal{E}\text{且}c_i(x)\geq 0, i\in \mathcal{I}\}.
$$

An obvious approach is to add the characteristic function of $\mathcal{X}$ to the objective function to form an unconstrained optimization problem, but the properties of the function are not very good. To solve this problem, we first need to give the optimality theory of constrained optimization problems. For this purpose, we introduceDefined as follows.

## Optimality conditions for constrained optimization problems
### Tangent cone

Given a feasible domain $\mathcal{X}$ and a point $x$ inside it, if there exists a feasible sequence $\{z_k\}_{k=1}^\infty\subset \mathcal{X}$ that satisfies $\lim_{k\to \infty}z_k = x$ and a positive scalar sequence $\{t_k\}_{k=1}^\infty$, $t_k\to 0$ satisfies
$$\lim\limits_{k\to \infty}\frac{z_k - x}{t_k} = d,$$
Then the vector $d$ is a tangent vector of $\mathcal{X}$ at $x$. The set of all tangent vectors at $x$ is called a **tangent cone**, denoted by $T_\mathcal{X}(x)$.

> In fact, the tangent cone here is the tangent space in differential geometry.

Similar to unconstrained optimization, we require that the tangent cone (a set of feasible directions) does not contain directions that make the objective function value decrease. This is a necessary condition that the local optimal point needs to meet, called $\textbf{geometric optimality condition}$.
### Geometric optimality condition
​ Assume that the feasible point $x^*\in\mathcal{X}$ is a local minimum point of the problem minimization problem. If$f(x)$ and $c_i(x)$, $i\in \mathcal{I}\cup \mathcal{E}$ are differentiable at point $x^*$, so we have

$$d^T\nabla f(x^*)\geq 0,\quad d\in T_\mathcal{X}(x^*)$$

Equivalent to

$$T_\mathcal{X}(x^*)\cap \{d\mid \nabla f(x^*)^Td<0\}=\emptyset$$

>The above theorem actually says that at the optimal point $x^*$, the intersection of the linearized feasible direction and the descending direction is an empty set, which is obvious.

### Active set
​ For a feasible point $x\in \mathcal{X}$, the $\textbf{active set}$ at this point is defined as a set of two indicators, one is the indicator corresponding to the equality constraint, and the other is the indicator corresponding to the constraint that holds in the inequality constraint at this point, that is,
$$\mathcal{A}(x) = \mathcal{E}\cup \{i\in\mathcal{I}:c_i = 0\}$$

### Linearly independent constraint quality LICQ
​ Given a feasible point $x\in\mathcal{X}$ and the corresponding active set $\mathcal{A}(x)$.If the gradient of the constraint function corresponding to the active set, i.e., $\nabla c_i(x), i\in\mathcal{A}(x)$, is linearly independent, then $\textbf{Linearly Independent Constraint Quality (LICQ)}$ is said to hold at point $x$.

### Linearized feasible direction cone
​ The \textbf{linearized feasible direction cone} at point $x$ is defined as
$$
\mathcal{F}(x) = \begin{cases}
d\mid \begin{split}
d^T\nabla c_i(x)=0,\forall i\in \mathcal{E},\\
d^T\nabla c_i(x)\geq 0,\forall i\in \mathcal{I}\cap \mathcal{A}(x)
\end{split}
\end{cases}
$$

> Lemma: Given any feasible point $x\in\mathcal{X}$, if LICQ holds at this point, then $T_\mathcal{X} = \mathcal{F}(x).$

It is still troublesome to directly verify that the intersection in the geometric optimality condition is an empty set. For this reason, we introduce a more convenient method

### Farkas Lemma

Let $p$ and $q$are two non-negative integers, given a set of vectors $\{a_i\in\mathbb{R}^n, i = 1,2,\cdots, p\}$, $\{b_i\in\mathbb{R}^n, i = 1,2,\cdots, q\}$ and $c\in\mathbb{R}^n$. The following conditions are satisfied:
$$d^Ta_i = 0,\quad i = 1, 2, \cdots, p,$$
$$d^Tb_i \geq 0,\quad i = 1, 2, \cdots, q,$$
$$d^Tc<0$$
$d$ does not exist if and only if there exists a set of $\lambda_i, i = 1,2,\cdots, p$ and $\mu_i\geq 0,i = 1,2, \cdots q$, so that
$$c = \sum\limits_{i=1}^p \lambda_ia_i + \sum\limits_{i=1}^q\mu_i b_i.$$

Using Farkas' lemma, we can write the geometric optimality condition (the intersection is an empty set) as the following equivalent form:
$$
-\nabla f(x^*) = \sum\limits_{i\in \mathcal{E}}\lambda_i^* \nabla c_i(x^*)+\sum\limits_{i\in \mathcal{A}(x^*)\cap\mathcal{I}}\lambda_i^*\nabla c_i(x^*),
$$
where $\lambda_i^*\in\mathbb{R}, i\in \mathcal{E},\lambda_i^*\geq 0, i\in \mathcal{A}(x^*)\cap\mathcal{I}$. If we define $\lambda_i^*=0,i\in \mathcal{I}\backslash\mathcal{A}(x^*)$, then we have
$$
-\nabla f(x^*) = \sum\limits_{i\in\mathcal{I}\cup\mathcal{E}}\lambda_i^*\nabla c_i(x^*),
$$
This happens to be the first-order optimality condition of the Lagrange function with respect to $x$. In addition, for any $i\in \mathcal{I}$, we note that
$$
\lambda_i^*c_i(x^*) = 0.
$$
The above formula is called the **complementary slack condition**, which means that for inequality constraints, at least one of the following two situations occurs:

1. The multiplier $\lambda_i\geq 0$
2. The inequality constraint fails, that is, $c_i(x^*>0)$ is strictly true

Generally speaking, if only one of the above two situations is satisfied, we call the **strict complementary relaxation condition** true. Generally speaking, the optimal value point with the strict complementary relaxation condition has better properties.

In summary, we have the following first-order necessary conditions, also called KKT conditions, and the point $x^*, \lambda^*$ that satisfies this condition is called a **KKT pair**.
### Karush-Kuhn-Tucker condition
Let $x^*$ be a local optimal point of the constrained optimization problem. If
$$T_{\mathcal{X}}(x^*) = \mathcal{F}(x^*)$$
holds, that is, LICQ holds at this point, then there exists a Lagrange multiplier $\lambda_i^*$ that makes the following conditions hold:

1. Stability condition $\nabla_x L(x^*, \lambda^*) = \nabla f(x^*)-\sum\limits_{i\in \mathcal{I}\cup \mathcal{E}}\lambda_i^*\nabla c_i(x^*) = 0$,
2. Original feasibility condition $c_i(x^*) = 0,\forall i\in\mathcal{E},$
3. Original feasibility condition $c_i(x^*) \geq 0,\forall i\in\mathcal{I},$
4. Dual feasibility condition $\lambda_i^*\geq 0,\forall i\in \mathcal{I}$,
5. Complementary relaxation condition $\lambda_i^*c_i(x^*) = 0,\forall i\in \mathcal{I}$

## Exercise
Find the KKT points and corresponding multipliers of the following constrained optimization problem
$$
\begin{split}
&\min f(x)=x_1^2+x_2\\
s.t. & -x_1^2-x_2^2+9\geq 0\\
& -x_1-x_2+1\geq 0
\end{split}
$$
Solution: 
The corresponding Lagrange function is
$$
\mathcal{L}(x,\lambda) = x_1^2+x_2 - \lambda_1 (-x_1^2-x_2^2+9) - \lambda_2 (-x_1-x_2+1)
$$
KKT condition is
$$
\begin{split}
&\frac{\partial\mathcal{L}}{\partial x_1} = 2x_1+2\lambda_1 x_1 +\lambda_2 = 0\\
&\frac{\partial\mathcal{L}}{\partial x_2} = 1+2\lambda_1 x_2 +\lambda_2 = 0\\
&\lambda_1 (-x_1^2-x_2^2+9) = 0, \lambda_1\geq 0, -x_1^2-x_2^2+9\geq 0\\
&\lambda_2 (-x_1-x_2+1) = 0, \lambda_2\geq 0, -x_1-x_2+1\geq 0
\end{split}
$$
First, $\lambda_1 = \lambda_2 = 0$ has no solution\\
$\lambda_1 = 0,\lambda_2 \neq 0$, we get $\lambda_2 = -1$, contradiction\\
$\lambda_1\neq 0, \lambda_2 = 0$, we get $x = (0, -3)^T,\lambda = (\frac{1}{6},0)^T$\\
$\lambda_1\neq 0, \lambda_2 \neq 0$, we get $x =(\frac{1\pm \sqrt{17}}{2},\frac{1\mp \sqrt{17}}{2})^T$, there is a contradiction of $\lambda_2 = -\frac{1}{2}$.

In summary, there is only one KKT point $(0, -3)^T$ and the corresponding Lagrange multiplier $(\frac{1}{6}, 0)^T$.