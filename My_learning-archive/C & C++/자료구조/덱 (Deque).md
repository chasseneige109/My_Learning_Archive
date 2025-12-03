## ✅ 2. `deque` – **양방향 큐(Deque = Double-Ended Queue)**

### ▶ 특징

- 앞(front)과 뒤(back) **양쪽 모두**에서 삽입/삭제 가능
    
- `queue`보다 유연함
    
- `vector`와 `list`의 장점을 결합한 구조 (중간 삽입은 느림)

|함수|설명|
|---|---|
|`dq.push_back(x)`|뒤에 `x` 추가|
|`dq.push_front(x)`|앞에 `x` 추가|
|`dq.pop_back()`|뒤 요소 제거|
|`dq.pop_front()`|앞 요소 제거|
|`dq.front()` / `dq.back()`|앞/뒤 요소 확인|
|`dq.empty()` / `dq.size()`|공백 / 크기 확인|

```
#include <iostream>
#include <deque>
using namespace std;

int main() {
    deque<int> dq;
    dq.push_back(10);
    dq.push_front(5);

    cout << dq.front() << '\n'; // 5
    cout << dq.back() << '\n';  // 10

    dq.pop_front(); // 5 제거
    cout << dq.front() << '\n'; // 10
}

```