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

### union by rank

### 경로 압축

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