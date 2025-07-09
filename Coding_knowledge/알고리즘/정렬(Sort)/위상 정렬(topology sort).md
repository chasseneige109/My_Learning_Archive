간선을 
```
for (int i = 1; i <= N; ++i) {
    for (int j = i + 1; j <= N; ++j) {
        int higher = last[i];
        int lower = last[j];
        adj[higher][lower] = true;
        indegree[lower]++;
```
이렇게 vec<vec<bool>> 배열로 구현.
int  indegree 라는 변수를 지정해서 진입차수를 저장시킴.






