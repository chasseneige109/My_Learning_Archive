
## ✅ 1. `+` 연산자 (문자열끼리 이어붙이기)

cpp

복사편집

`string a = "Hello"; string b = "World"; string c = a + " " + b;  // "Hello World"`

---

## ✅ 2. `+=` 연산자 (뒤에 덧붙이기)

cpp

복사편집

`string s = "Hello"; s += " World";     // "Hello World" s += '!';          // "Hello World!"`

---

## ✅ 3. `append()` 함수

cpp

복사편집

`string s = "Hi"; s.append(" there");        // "Hi there" s.append(3, '!');          // "Hi there!!!" (문자 3개 붙임)`

---

## ✅ 4. `push_back(char)` — 문자 1개만

cpp

복사편집

`string s = "A"; s.push_back('B');  // "AB"`