| 외부 함수             | **입력 조건**         | **출력 성질** | **예시**                                    | **허용 여부** |
| ----------------- | ----------------- | --------- | ----------------------------------------- | --------- |
| **Affine**        | 아무거나 가능           | 같은 성질 유지  | `2*x + y`                                 | ✅         |
| **Convex**        | **Affine 입력**만 허용 | Convex    | `square(x+y)`                             | ✅         |
| **Convex & 비감소**  | Convex 입력 가능      | Convex    | `exp(norm(x))`                            | ✅         |
| **Convex & 비증가**  | Concave 입력 가능     | Convex    | `-log(y)` (y>0, log concave, -log convex) | ✅         |
| **Concave**       | **Affine 입력**만 허용 | Concave   | `sqrt(x+y)`                               | ✅         |
| **Concave & 비증가** | Convex 입력 가능      | Concave   | `-sqrt(x)`                                | ✅         |
| **Concave & 비감소** | Concave 입력 가능     | Concave   | `log(x)`                                  | ✅         |
