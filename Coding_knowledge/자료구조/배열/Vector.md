### ✅ 기본 선언 방법

```
#include <vector> 
using namespace std;  
vector<int> v;  // int형 빈 벡터
```

---

### ✅ 초기 크기와 기본값 설정

```
vector<int> v(5);        // 크기 5, 기본값 0 
vector<int> v(5, 7);     // 크기 5, 모든 값 7
```

---

### ✅ 배열 초기화 방식

```
vector<int> v = {1, 2, 3, 4};   // 초기값 1,2,3,4
```

### ✅ 2차원 벡터

```
vector<vector<int>> mat(3, vector<int>(4, 0));  // 3x4, 모두 0
```

### ✅ 원소 접근 (`[]`, `at()`)

```
cout << v[0];      // 0번째 원소 
cout << v.at(1);   // 1번째 원소 (범위 검사함)
```