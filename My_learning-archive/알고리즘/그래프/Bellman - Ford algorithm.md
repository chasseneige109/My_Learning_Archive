
# 🧠 벨만-포드 알고리즘 (Bellman-Ford Algorithm)

## ✅ 1. 언제 쓰나?

- **단일 시작점 → 모든 정점까지 최단 거리 구할 때**
    
- 간선에 **음수 가중치가 있을 수 있을 때**
    
- **음의 사이클 유무도 판별 가능**함
    

> 다익스트라는 음수 간선 ❌ 못 다룸 → 벨만-포드는 가능 ⭕

---

## ✅ 2. 핵심 아이디어

> **모든 간선을 N-1번 반복해서 Relaxation(완화)** 한다.

즉:

- 각 정점까지의 거리를 저장하는 배열 `dist[]`을 유지
    
- 매 반복마다, **모든 간선(u → v, w)**에 대해
    
    cpp
    
    복사편집
    
    `if (dist[u] + w < dist[v]) {     dist[v] = dist[u] + w; }`
    
    이걸 N-1번 반복함
    

### ❓ 왜 N-1번?

- 그래프에 정점이 N개면, 최악의 경우 최단 경로는 **N-1개의 간선**으로 구성됨
    
- 즉, **최단 경로는 최대 N-1번의 간선 relax로 만들어짐**
    

---

## ✅ 3. 음수 사이클 검출

N번째에 한 번 더 돌렸을 때 **갱신이 일어나면** → **음의 사이클 존재**

cpp

복사편집

`if (i == N && dist[u] + w < dist[v]) {     // 음의 사이클 존재 }`

---

## ✅ 4. 구현 예시 (C++)

cpp

복사편집

```
`int N, M; // 정점 수, 간선 수 vector<tuple<int, int, int>> edges; // 간선 (u, v, w) vector<long long> dist(N + 1, INF); dist[1] = 0;  bool negative_cycle = false;  for (int i = 1; i <= N; ++i) {     for (auto [u, v, w] : edges) {         if (dist[u] != INF && dist[v] > dist[u] + w) {             dist[v] = dist[u] + w;             if (i == N) negative_cycle = true;         }     } }`
```

---

## ✅ 5. 시간복잡도

- **O(N × M)**  
    (정점 수 × 간선 수)
    

> ✔️ N = 500, M = 6000 → 3,000,000 → 충분히 가능