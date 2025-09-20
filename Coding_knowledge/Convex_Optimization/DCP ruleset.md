
# Table

| **Outer function**          | **Allowed input** | **Resulting function** | **Example**     | **Valid?** |
| --------------------------- | ----------------- | ---------------------- | --------------- | ---------- |
| **Affine**                  | Anything          | Same as input          | `2*x + y`       | ✅          |
| **Convex**                  | Affine only       | Convex                 | `square(x+y)`   | ✅          |
| **Convex & nondecreasing**  | Convex input      | Convex                 | `exp(norm(x))`  | ✅          |
| **Convex & nonincreasing**  | Concave input     | Convex                 | `-log(y)` (y>0) | ✅          |
| **Concave**                 | Affine only       | Concave                | `sqrt(x+y)`     | ✅          |
| **Concave & nondecreasing** | Concave input     | Concave                | `log(x)`        | ✅          |
| **Concave & nonincreasing** | Convex input      | Concave                | `-sqrt(x)`      | ✅          |

# Some Special Rules & Cases

when f(X) is convex,
1. square(f(X)) is convex when f(X) >= 0
2. 

# Examples from the lecture

1. norm (x + 2y, x - y) == 0   (X)
2. square(square(x+y)) (X)
   - variable t 
   - square( x+y ) <= t 
   - square( t ) <= x- y
3. 