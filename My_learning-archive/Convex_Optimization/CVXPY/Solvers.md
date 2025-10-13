## ✅ CVXPY Solver 

|Solver 이름|문제 유형|특징|비고|
|---|---|---|---|
|**ECOS**|LP, QP, SOCP|빠르고 가벼움 (기본값 중 하나)|CVXPY에 기본 포함|
|**SCS**|LP, QP, SOCP, EXP, SDP|**대규모 문제에 강함**, 정확도 낮을 수 있음|기본 포함|
|**OSQP**|QP|**Quadratic Programming 특화**, 매우 빠름|기본 포함|
|**GLPK**|LP|단순 선형 문제(LP)용|기본 포함|
|**GLPK_MI**|MILP|정수/이진 변수 포함 문제|Mixed-Integer LP|
|**CBC**|LP, MILP|오픈소스 Mixed Integer 솔버|설치 필요|
|**GUROBI**|LP, QP, QCP, MIP|상용 솔버, 매우 빠름|별도 라이선스 필요|
|**CPLEX**|LP, QP, QCP, MIP|상용 솔버 (IBM)|설치 필요|
|**MOSEK**|LP, QP, SOCP, SDP, EXP|고성능 상용 솔버, 정확도 높음|연구용 무료 라이선스 있음|
|**CVXOPT**|LP, QP|Python 기반 솔버|작고 간단함|
|**XPRESS**|LP, QP, MIP|상용 솔버 (FICO)|설치 필요|
|**NAG**|LP, QP, SOCP|상용 솔버 (NAG)|설치 필요|