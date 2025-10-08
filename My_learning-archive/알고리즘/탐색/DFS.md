
## ✅ DFS 구현 with 스택 vs 재귀 vs BFS

DFS 재귀 vs 스택 vs BFS: 언제 무엇을 쓸까?

#### **재귀 DFS**
- "밑에서부터 빠져나오면서" POST-ORDER 로직 실행 
#### **스택 DFS**
- PRE-ORDER만 가능해서 'BFS'와 본질적으로 비슷함
#### BFS
- 스택 DFS보다는 나은 것

### 🔁 왜 DFS 스택보다 BFS가 나은가?

#### **1. 구현 복잡도**: 비슷함

#### **2. 메모리 사용**:

- DFS 스택: O(깊이)
- BFS: O(너비)
- 대부분 상황에서 BFS가 더 예측 가능

#### **3. 디버깅**: BFS가 더 직관적 (레벨별 진행)

#### **4. 확장성**: BFS는 가중치 그래프로 확장 용이 (다익스트라)


### 🔁 언제 DFS 스택을 쓸까?

정말 특수한 경우에만:

- 재귀 깊이 > 10만 (스택 오버플로우)
- 메모리가 극도로 제한적
- **그런데 이런 경우에도 BFS를 먼저 고려해볼 것!**

당신 말이 맞습니다. **DFS 스택은 사실상 "재귀를 못 쓸 때의 차선책"**이고, 그런 상황이라면 **BFS를 먼저 고려하는 게 현명**합니다!




# 리프 노드 먼저 처리하며 순회하는 DFS

## ✅ 방법 1: Post-order DFS

### (가장 일반적인 방식)

```
void dfs(int node, int parent) {
    for (int child : tree[node]) {
        if (child != parent) {
            dfs(child, node);
        }
    }
    // 자식들을 다 본 후, 현재 노드 처리 (가장 깊은 노드부터 올라오는 순서)
    doSomething(node);
}

```

이 방식은 **리프 → 부모 → 조상 순**으로 진행돼서 "깊은 노드부터 순회"에 해당함.

---

## ✅ 방법 2: DFS로 depth 기록 + 정렬

1. 각 노드의 `depth`를 기록
    
2. `depth` 기준으로 내림차순 정렬
    
3. 정렬된 순서대로 처리
    
```
vector<int> order;
int depth[100001];

void dfs(int node, int parent, int d) {
    depth[node] = d;
    for (int child : tree[node]) {
        if (child != parent)
            dfs(child, node, d + 1);
    }
    order.push_back(node); // DFS 완료 후 넣으면 post-order
}

...

dfs(root, -1, 0);
sort(order.begin(), order.end(), [](int a, int b) {
    return depth[a] > depth[b]; // 깊은 노드부터
});

for (int node : order)
    doSomething(node);

```

---

## ✅ 언제 이런 방식 쓰냐?

- **리프 노드부터 부모 노드로 값을 누적**해야 할 때  
    (ex: 서브트리 합, 자식 개수 계산, DP 등)
    
- **트리 DP** 문제에서 자주 등장함