## ✅ **Top-down (Memoization)**

- **재귀 + 메모이제이션**
    
- 하위 문제의 답을 저장하고 필요할 때만 계산
    
- 보통 코드가 짧고 직관적이지만, **스택 오버플로우** 주의

```
int dp[1001];

int fib(int n) {
    if (n <= 1) return n;
    if (dp[n] != -1) return dp[n];
    return dp[n] = fib(n-1) + fib(n-2);
}

```