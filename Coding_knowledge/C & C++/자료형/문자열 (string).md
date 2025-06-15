## ✅ string 뒤에 이어붙이기 
###  1. `+` 연산자 (문자열끼리 이어붙이기)


a + b; 

---

###  2. `+=` 연산자 (뒤에 덧붙이기)


`string s = "Hello"; s += " World";

---

###  3. `append()` 함수


```
string s = "Hi"; 
s.append(" there");        
// "Hi there" 
s.append(3, '!');         
// "Hi there!!!" (문자 3개 붙임)
```

---

###  4. `push_back(char)` — 문자 1개만

```
string s = "A"; 
s.push_back('B');  
// "AB"
```

### 5. 추출

|함수|설명|
|---|---|
|`substr(pos, len)`|pos부터 len개 추출한 새 문자열 반환|

| 함수                 | 설명                            |     |     |     |
| ------------------ | ----------------------------- | --- | --- | --- |
| `compare(str)`     | 사전순 비교 (0: 같음)                |     |     |     |
| `find(str)`        | str의 첫 위치 인덱스 반환 (없으면 `npos`) |     |     |     |
| `rfind(str)`       | str의 마지막 위치 인덱스 반환            |     |     |     |
| `starts_with(str)` | (C++20) 시작 문자열 확인             |     |     |     |
| `ends_with(str)`   | (C++20) 끝 문자열 확인              |     | 함수  | 설명  |
| `push_back(char)`  | 맨 뒤에 문자 추가                    |     |     |     |
| `pop_back()`       | 맨 뒤 문자 제거                     |     |     |     |
| `append(str)`      | 문자열 끝에 붙이기                    |     |     |     |
| `insert(pos, str)` | 특정 위치에 문자열 삽입                 |     |     |     |
| `erase(pos, len)`  | 특정 위치부터 len개 문자 제거            |     |     |     |
| `clear()`          | 문자열 전체 삭제                     |     |     |     |
| 함수                 | 설명                            |     |     |     |
| `operator[]`       | 특정 인덱스의 문자 반환                 |     |     |     |
| `at(index)`        | 인덱스의 문자 반환 (범위 체크 O)          |     |     |     |
| `front()`          | 첫 문자 반환                       |     |     |     |
| `back()`           | 마지막 문자 반환                     |     |     |     |