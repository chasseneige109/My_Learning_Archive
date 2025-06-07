### ✅ 기본 개념

`#include <map>`

`map<KeyType, ValueType> map이름;`

---

### ✅ 특징 요약

|항목|설명|
|---|---|
|정렬|키 기준 **오름차순** 자동 정렬 (`<` 연산자 기준)|
|중복 키|❌ 허용되지 않음|
|삽입 시간|O(log n) (이진 트리 기반)|
|내부 구조|`Red-Black Tree` (균형 이진 탐색 트리)|
|반복자|정렬된 순서로 접근 가능|

---

### ✅ 기본 사용 예시


`#include <iostream> #include <map> using namespace std;  int main() {     map<string, int> score;      // 삽입     score["Alice"] = 90;     score["Bob"] = 85;     score["Charlie"] = 95;      // 검색     cout << score["Bob"] << endl; // 85      // 반복 출력     for (auto it : score) {         cout << it.first << " : " << it.second << endl;     }      return 0; }`

출력:

yaml

복사편집

`85 Alice : 90 Bob : 85 Charlie : 95`

---

### ✅ 주요 함수/연산자

|함수/연산|설명|
|---|---|
|`m[key]`|값 접근/삽입 (없으면 자동 생성)|
|`m.at(key)`|값 접근 (없으면 예외 발생)|
|`m.insert({key, val})`|삽입 (중복 시 무시)|
|`m.erase(key)`|삭제|
|`m.find(key)`|반복자 반환 (없으면 `end()`)|
|`m.count(key)`|존재 여부 확인 (0 또는 1)|
|`m.clear()`|전체 삭제|
|`m.size()`|요소 수 반환|

---

### ✅ 기타

- `unordered_map`은 해시 기반으로 `정렬 안 됨` + `더 빠른 평균 성능`
    
- `multimap`은 `중복 키 허용`