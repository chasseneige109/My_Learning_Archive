## ✅ `min` 함수 사용 방법

### 🔍 문법
```
#include <algorithm> // 이게 있어야 함

int a = 5, b = 10;
int smaller = std::min(a, b); // smaller == 5
```

### ❗주의

- `min`은 매크로나 함수로 정의되어 있어서 **템플릿 인자 타입이 같아야** 함.
    
- `min(a, b)`에서 `a`, `b`가 서로 다른 타입이면 **컴파일 에러**가 날 수 있어.
    
    - 해결: `min(static_cast<int>(a), b)` 같이 타입을 맞춰줌


## ✅ min 3개 이상일 경우

### 🔍방법 1: `std::min` 중첩 사용



