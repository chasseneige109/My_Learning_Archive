## ✅ 1. `queue` – **선입선출(FIFO)** 자료구조

### ▶ 특징

- 앞(front)에서 꺼내고, 뒤(back)에 넣음
    
- 오직 **한쪽에서 넣고**, **한쪽에서만 꺼냄**
    
- 내부적으로 `deque`나 `list`를 기반으로 구현됨
    

### ▶ 주요 함수

|함수|반환형|설명|
|---|---|---|
|`q.push(const T& x)`|`void`|뒤에 요소 `x` 삽입|
|`q.push(T&& x)`|`void`|이동 시맨틱으로 `x` 삽입 (`C++11~`)|
|`q.pop()`|`void`|앞(front) 요소 제거 (반환 없음)|
|`q.front()`|`T&` / `const T&`|맨 앞 요소 반환|
|`q.back()`|`T&` / `const T&`|맨 뒤 요소 반환|
|`q.empty()`|`bool`|큐가 비었는지 확인 (`true`/`false`)|
|`q.size()`|`size_t`|큐의 요소 개수 반환|
|`q.emplace(args...)`|`void`|**요소를 직접 생성해서 삽입** (`C++11~`)|

### ▶ 코드 예시


```
#include <iostream> 
#include <queue> 
using namespace std; 

int main() {     

queue<int> q;     

q.push(10);     
q.push(20);      

cout << q.front() << '\n'; // 10     
cout << q.back() << '\n';  // 20     

q.pop();     

cout << q.front() << '\n'; // 20 
}
```