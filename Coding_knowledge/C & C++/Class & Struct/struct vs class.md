
# ✅ `struct` vs `class` 핵심 차이

|항목|`struct`|`class`|
|---|---|---|
|**기본 접근제한자**|`public`|`private`|
|사용 목적|**데이터 묶음** (structural)|**캡슐화 / 객체 설계**|
|연산자 오버로딩|가능|가능|
|상속|가능|가능|
|실무 사용|POD (Plain Old Data), 알고리즘, STL|클래스, 캡슐화, OOP 설계|

## 🎯 실전 기준 선택 기준

| 상황               | 추천                      |
| ---------------- | ----------------------- |
| 알고리즘 / STL 자료 구조 | `struct` (간단하고 빠름)      |
| 클래스 설계 / 캡슐화 필요  | `class` (OOP 기능 활용)     |
| 비교 연산자만 필요       | `struct` + operator<    |
| 접근 제한 / 은닉 필요    | `class` + getter/setter |

## ✅ STL도 사실 `struct`임


`template<class T, class Container = deque<T>> struct queue;`

👉 `queue`, `pair`, `tuple` 전부 `struct`임!