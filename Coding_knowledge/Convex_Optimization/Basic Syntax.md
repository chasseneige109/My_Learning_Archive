
| 범주                   | 함수 / 문법                      | 설명 (한글)                   | 설명 (영문)                               |
| -------------------- | ---------------------------- | ------------------------- | ------------------------------------- |
| **변수 정의**            | `cp.Variable()`              | 스칼라 변수                    | Scalar variable                       |
|                      | `cp.Variable(n)`             | 길이 n 벡터 변수                | Vector variable (length n)            |
|                      | `cp.Variable((m,n))`         | m×n 행렬 변수                 | Matrix variable                       |
|                      | `cp.Variable(nonneg=True)`   | 0 이상 제약 포함                | Nonnegative variable                  |
|                      | `cp.Variable(boolean=True)`  | 이진 변수 (0/1)               | Boolean variable                      |
|                      | `cp.Variable(integer=True)`  | 정수 변수                     | Integer variable                      |
| **문제 정의**            | `cp.Minimize(expr)`          | expr 최소화                  | Minimize objective                    |
|                      | `cp.Maximize(expr)`          | expr 최대화                  | Maximize objective                    |
|                      | `cp.Problem(obj, cons)`      | 문제 정의                     | Define optimization problem           |
|                      | `.solve()`                   | 문제 풀기                     | Solve the problem                     |
| **목적함수/제약**          | `x >= 0`                     | x는 0 이상                   | Nonnegativity constraint              |
|                      | `A @ x <= b`                 | 선형 부등식                    | Linear inequality                     |
|                      | `C @ x == d`                 | 선형 등식                     | Linear equality                       |
|                      | `X >> 0`                     | X는 PSD 행렬                 | Positive semidefinite matrix          |
| **Norms**            | `cp.norm(x,1)`               | L1 norm (‖x‖₁)            | L1 norm                               |
|                      | `cp.norm(x,2)`               | L2 norm (‖x‖₂)            | L2 norm                               |
|                      | `cp.norm(x,"inf")`           | 무한 norm (max)             | Infinity norm                         |
| **원소 연산**            | `cp.abs(x)`                  | 절댓값                       | Absolute value                        |
|                      | `cp.square(x)`               | 제곱                        | Square                                |
|                      | `cp.sqrt(x)`                 | 제곱근                       | Square root                           |
|                      | `cp.multiply(x,y)`           | 원소별 곱                     | Elementwise multiply                  |
|                      | `cp.maximum(x,y)`            | 원소별 최대                    | Elementwise max                       |
|                      | `cp.minimum(x,y)`            | 원소별 최소                    | Elementwise min                       |
| **행렬 함수**            | `cp.trace(X)`                | Trace                     | Matrix trace                          |
|                      | `cp.sum(X, axis=0/1)`        | 열/행 합                     | Sum by axis                           |
|                      | `cp.reshape(x,(m,n))`        | 모양 변경                     | Reshape                               |
|                      | `cp.normNuc(X)`              | Nuclear norm              | Nuclear norm (sum of singular values) |
| **통계/평균**            | `cp.sum(x)`                  | 합                         | Sum                                   |
|                      | `cp.mean(x)`                 | 평균                        | Mean                                  |
|                      | `cp.variance(x)`             | 분산                        | Variance                              |
|                      | `cp.geo_mean(x)`             | 기하평균                      | Geometric mean                        |
| **특수 함수**            | `cp.inv_pos(x)`              | 1/x (x>0)                 | Reciprocal                            |
|                      | `cp.log(x)`                  | log                       | Natural logarithm                     |
|                      | `cp.exp(x)`                  | exp                       | Exponential                           |
|                      | `cp.kl_div(x,y)`             | KL 발산                     | Kullback–Leibler divergence           |
|                      | `cp.huber(x,M=1)`            | Huber 손실                  | Huber loss function                   |
| **Quadratic/Linear** | `cp.quad_form(x,P)`          | xᵀPx                      | Quadratic form                        |
|                      | `A @ x - b`                  | 선형 residual               | Linear residual                       |
| **Solver 옵션**        | `prob.solve(solver=cp.SCS)`  | SCS solver 사용             | Use SCS solver                        |
|                      | `prob.solve(solver=cp.ECOS)` | ECOS solver 사용            | Use ECOS solver                       |
|                      | `prob.solve(verbose=True)`   | 로그 출력                     | Verbose logging                       |
| **결과 확인**            | `prob.status`                | 상태: optimal, infeasible 등 | Problem status                        |
|                      | `prob.value`                 | 최적 목적함수 값                 | Optimal objective value               |
|                      | `x.value`                    | 변수 최적해                    | Optimal variable value                |
|                      | `constraint.dual_value`      | 제약의 dual 변수 λ             | Dual variable of constraint           |