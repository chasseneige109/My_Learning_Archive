# 🧮 Schur Complement 정리

---

## 1️⃣ 기본 정의

대칭 행렬 $X = X^T \in \mathbb{R}^{n \times n}$ 이 다음처럼 블록으로 나누어져 있다고 하자.

$$
X =
\begin{bmatrix}
A & B \\
B^T & C
\end{bmatrix},
\qquad
A \in \mathbb{R}^{k\times k}, \;
C \in \mathbb{R}^{(n-k)\times (n-k)}.
$$

**Schur complement of $A$ in $X$** 는 다음과 같이 정의된다.

$$
S = C - B^T A^{-1} B,
$$

단, $A$가 **invertible**할 때만 유효하다.

---

## 2️⃣ 직관적 의미

$f(u,v)$를 quadratic form으로 두면,

$$
f(u,v) =
\begin{bmatrix} u^T & v^T \end{bmatrix}
\begin{bmatrix} A & B \\ B^T & C \end{bmatrix}
\begin{bmatrix} u \\ v \end{bmatrix}
= u^T A u + 2 v^T B^T u + v^T C v.
$$

$A \succ 0$일 때, $u$에 대해 최소화하면

$$
u^*(v) = -A^{-1}Bv,
$$

이를 대입하면

$$
g(v) = f(u^*(v), v) = v^T (C - B^T A^{-1}B) v = v^T S v.
$$

즉, **한 변수($u$)** 를 제거했을 때 남는 **조건부 안정성**(또는 조건부 공분산)이 Schur complement다.

---

## 3️⃣ PSD 조건과 Schur Complement

$$
X \succeq 0
\iff
\begin{cases}
A \succeq 0,\\
B^T(I - A A^\dagger) = 0,\\
C - B^T A^\dagger B \succeq 0.
\end{cases}
$$

- $A^\dagger$: Moore–Penrose pseudoinverse  
- $I - A A^\dagger$: $A$의 **null space**로의 projection

---

### 💬 해석

| 상황 | 의미 |
|------|------|
| $A \succ 0$ | 모든 방향에서 안정성 확보 → 추가 제약 불필요 |
| $A \succeq 0$ | 일부 null space 존재 → 그 방향에서 $u^T A u = 0$, 교차항이 위험함 |
| 제약 $B^T(I - A A^\dagger) = 0$ | $B$가 $A$의 null space 방향으로 성분을 가지면 안 됨 |
| $C - B^T A^\dagger B \succeq 0$ | “조건부 부분”이 양의 반정부호일 것 |

---

## 4️⃣ Schur Complement의 수학적 특징

| 성질 | 식 |
|------|----|
| Determinant 분해 | $\det X = \det A \cdot \det(C - B^T A^{-1} B)$ |
| Inverse 분해 | $X^{-1} = \begin{bmatrix} * & * \\ * & (C - B^T A^{-1}B)^{-1} \end{bmatrix}$ |
| PSD 조건 | $X \succeq 0 \iff A \succeq 0,\; B^T(I-AA^\dagger)=0,\; C-B^T A^\dagger B\succeq 0$ |

---

## 5️⃣ 왜 Block으로 나누는가?

$X$는 단순히 symmetric matrix지만,  
block 분할을 통해 **부분 공간 간의 안정성 / 상호작용**을 따로 볼 수 있다.

$$
z = \begin{bmatrix}u \\ v\end{bmatrix}, \quad
z^T X z = u^T A u + 2 v^T B^T u + v^T C v
$$

- $A$: $u$-subspace의 안정성  
- $C$: $v$-subspace의 안정성  
- $B$: 두 공간 간 coupling (상호작용)

이를 통해 “한 공간을 제거한 후 남는 공간의 조건부 안정성”을 정량적으로 분석할 수 있다.

---

## 6️⃣ 실무적 활용 예시

| 분야 | $u$ / $v$ 구분 | Schur complement의 역할 |
|------|----------------|--------------------------|
| **Convex Optimization / SDP** | 일부 변수 제거 | 부분 변수 제거 (partial minimization) |
| **Statistics / ML** | 관측 변수 $u$ vs 숨은 변수 $v$ | 조건부 공분산 $\Sigma_{v|u} = \Sigma_{vv} - \Sigma_{vu}\Sigma_{uu}^{-1}\Sigma_{uv}$ |
| **Control Theory (LMI)** | subsystem 1 vs subsystem 2 | subsystem 안정성 보장 조건 |
| **Robotics / SLAM** | 오래된 state vs 새 state | marginalization (과거 state 제거 후 covariance 갱신) |
| **Structural Mechanics** | 고정점 vs 자유점 | 하부 구조 제거 후 stiffness 갱신 |
| **Numerical Linear Algebra** | block elimination | 효율적인 부분 해 계산 |
| **전기회로 해석** | 고정 노드 vs 자유 노드 | reduced admittance matrix 계산 |

---

## 7️⃣ 직관 요약

> Schur complement는  
> “큰 행렬의 PSD(안정성) 조건을 부분 변수 단위로 분리하여,  
> 한 변수를 제거했을 때 남는 부분이 여전히 안정한가”를 분석하는 도구다.

---

## 8️⃣ 시각적 직관

- $A$: “u 영역 내부의 구조”
- $C$: “v 영역 내부의 구조”
- $B$: “u ↔ v 경계 coupling”

가령 $u$가 “가로수 이전의 공간”, $v$가 “가로수 이후의 공간”이라면  
$S = C - B^T A^{-1} B$는 **“이전 구간(u)을 고려한 뒤 이후 구간(v)의 조건부 안정성”** 을 나타낸다.

---

## ✅ 핵심 요약

- Schur complement: $S = C - B^T A^{-1} B$
- Pseudoinverse 버전: $S = C - B^T A^\dagger B$, 단 $B^T(I - A A^\dagger) = 0$
- $A \succ 0$이면 단순한 형태로 충분,  
  $A \succeq 0$이면 null space 보정 필요
- 실무에서는 subsystem 분석, 조건부 안정성, marginalization 등에 폭넓게 사용됨
