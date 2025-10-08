### 설명

- **정의**: string::npos는 std::string::size_type 타입의 상수로, 보통 -1을 unsigned integer로 표현한 값입니다 (예: std::string::size_type은 unsigned long 또는 size_t로 구현됨). 즉, 매우 큰 양수 값(예: 2^64-1)입니다.

- **용도**: 문자열에서 특정 패턴이나 문자를 찾으려고 했지만, 찾지 못했을 때 find 계열 함수가 string::npos를 반환합니다.

- 예시
```
if (to_string(num).find("666") != string::npos)
{
	count++;
}
```