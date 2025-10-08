### 기본적인 구현 방법 
```
for (int i = 1; i <= N; ++i) {
    for (int j = i + 1; j <= N; ++j) {
        int higher = last[i];
        int lower = last[j];
        adj[higher][lower] = true;
        indegree[lower]++;
```
간선을 vec<vec<bool>> 배열로 구현.
int  indegree 라는 변수를 지정해서 진입차수(해당 노드에게 들어오는 화살표 개수)를 저장시킴.

이후 pq에 indegree == 0인 경우부터 넣고 BFS로 정렬.

```
while (!pq.empty()) {
    int curr = pq.top();
    pq.pop();
    cout << curr << ' ';

    for (int next : graph[curr]) {
        if (--indegree[next] == 0) {
            pq.push(next);
        }
    }
}
```


DFS도있음







