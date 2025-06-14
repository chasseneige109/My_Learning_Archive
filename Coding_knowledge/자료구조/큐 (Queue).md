## ✅ 1. `queue` – **선입선출(FIFO)** 자료구조

### ▶ 특징

- 앞(front)에서 꺼내고, 뒤(back)에 넣음
    
- 오직 **한쪽에서 넣고**, **한쪽에서만 꺼냄**
    
- 내부적으로 `deque`나 `list`를 기반으로 구현됨
    

### ▶ 주요 함수

|함수|설명|
|---|---|
|`q.push(x)`|뒤에 `x` 추가|
|`q.pop()`|앞에서 요소 제거|
|`q.front()`|가장 앞의 요소 확인|
|`q.back()`|가장 뒤의 요소 확인|
|`q.empty()`|비었는지 확인|
|`q.size()`|요소 개수 확인|

### ▶ 코드 예시

cpp

복사편집

`#include <iostream> #include <queue> using namespace std;  int main() {     queue<int> q;     q.push(10);     q.push(20);      cout << q.front() << '\n'; // 10     cout << q.back() << '\n';  // 20     q.pop();     cout << q.front() << '\n'; // 20 }`