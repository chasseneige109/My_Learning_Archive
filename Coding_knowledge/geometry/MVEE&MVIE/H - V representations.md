## MVEE / MVIE 둘다

✅ **V-representation (points given)**  
→ convex problem → logdet-min, CVXPY로 풀림.

❌ **H-representation (halfspaces given)**  
→ “모든 x ∈ polytope” 조건 때문에 hidden max 존재 → NP-hard.  
→ 근사 알고리즘 (e.g. sampling, outer-approximation, dual relaxation)으로만 가능.