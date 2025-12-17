# H-representation -> V-representation 변환

---

## 과정 (3D 예시)

예를 들어 3차원에서 다면체가

Ax≤bA x \le bAx≤b

로 주어졌다고 합시다.  
각 행 aiTx=bia_i^T x = b_iaiT​x=bi​는 **하나의 면(face)** 을 정의하죠.

**faces to vertices method**는 대략 다음 절차로 진행됩니다:

1. **면 3개씩 골라서 교점 계산**  
    → ai,aj,aka_i, a_j, a_kai​,aj​,ak​ 세 개의 평면을 풀면 교점 하나 xijkx_{ijk}xijk​ 얻음.
    
2. **그 교점이 모든 부등식 Ax≤bA x \le bAx≤b를 만족하는지 확인**  
    → 즉, 다면체 내부인지 검사.
    
3. **모든 면 조합에 대해 반복**, 내부 교점만 남김.
    
4. **중복 제거 및 정렬**  
    → 실제 꼭짓점(vertex) 집합 완성.
    

이게 기본적인 **faces → vertices 변환 알고리즘**이에요.

* CVXPY는 기본적으로 H-representation (Ax ≤ b) 기반으로 동작