int find(int x)
{
	if (parent[x] == x)
		return x;
	else
		return parent[x] = find(parent[x]);
}

void uni (int x, int y)
{
	x = find(x);
	y = find(y);
	if (x != y)
	{
		parent[y] = x;
	}
}
# ✅ 1. 경로 압축 (Path Compression)

### 🧠 개념:

- `find(x)`를 호출할 때, x가 루트가 아니면 **그 부모를 루트로 직접 연결**해버리는 방식
    
- → 다음 `find(x)`는 O(1)처럼 작동
# ✅ 2. 랭크 기반 유니온 (Union by Rank or Size)

### 🧠 개념:

- 집합 병합 시, **작은 트리를 큰 트리에 붙인다**
    
- 트리의 높이(또는 노드 수)를 최소화해 성능을 향상시킴


# 💻 전체 코드 예시: Union by Rank + Path Compression
```
int find(int x) {
    if (parent[x] == x)
        return x;
    return parent[x] = find(parent[x]); // 경로 압축
}

void unionSet(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return;

    if (rank[a] < rank[b]) {
        parent[a] = b;
    } else {
        parent[b] = a;
        if (rank[a] == rank[b])
            rank[a]++;
    }
}
```


# 그룹의 개수

netsize배열 추가해서 uni에서 parnet