#include <iostream>
#include <set>
using namespace std;

int main() {
    set<int> s;
    s.insert(5);
    s.insert(3);
    s.insert(8);
    s.insert(5);  // 중복 → 무시됨

### 🔍 특징

- **중복을 허용하지 않음**
    
- **자동 정렬됨 (기본: 오름차순)**
    
- `insert()`, `erase()`, `find()` 등 함수 사용
    
- 내부는 이진 탐색 트리(Red-Black Tree) 기반 → 시간 복잡도 O(log N)
    
|함수|설명|
|---|---|
|`insert(value)`|값 추가 (중복 시 무시됨)|
|`erase(value)`|값 제거|
|`find(value)`|값 찾기 (없으면 `end()` 반환)|
|`count(value)`|값의 개수 (set은 0 또는 1만 나옴)|
|`size()`|전체 원소 개수|
|`clear()`|전부 삭제|
|`empty()`|비었는지 확인|
|`begin(), end()`|반복자 (iterator)|