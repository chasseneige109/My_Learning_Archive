
## 🔹 다익스트라 알고리즘 구현법 (C++ 기준)

- **조건**: 양의 가중치 그래프
    
- **목적**: 시작 정점 → 모든 정점 최단거리 구하기 (DP)

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