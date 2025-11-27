
### 계산 순서
t=0:  x0 → h0 → y0
t=1:  x1 → h1 → y1
t=2:  x2 → h2 → y2
     ...

직렬성이 강함. 병렬성이 떨어짐. GPU가 노는 시간이 생김

h(t - 1) -> h(t)와 x(t) -> h(t) affine 변환 둘이 합쳐져서 Z를 만들고 activation을 지나 H가 됨.


### shared weight
- 시간에 따라 layer는 여러 개지만
    
- 모든 layer가 **동일한 파라미터를 공유함**


