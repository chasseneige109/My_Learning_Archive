간선의 가중치가 존재하며, 한 노드에서 그래프 내에 존재하는 모든 다른 한 노드로 가는 최소비용을 Dynamic programming을 통해 한 번에 구하는 알고리즘.


```
vector<vector<pair<int, int>>> graph(V + 1);  // graph[u] = {{v, w}, ...}
vector<int> dist(V + 1, INF);
priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

dist[start] = 0;
pq.emplace(0, start);

while (!pq.empty()) {
    auto [cost, curr] = pq.top(); pq.pop();
    if (dist[curr] < cost) continue;

    for (auto [next, weight] : graph[curr]) {
        if (dist[next] > cost + weight) {
            dist[next] = cost + weight;
            pq.emplace(dist[next], next);
        }
    }
}

```
## 📌 요약 키워드

- 인접 리스트 (`vec<vec<pair>>`)
    
- `dist[]`: 최소 거리 저장
    
- `priority_queue`: 가장 짧은 거리 정점 먼저 탐색
    
- 갱신 조건: `if (dist[next] > dist[curr] + w)`