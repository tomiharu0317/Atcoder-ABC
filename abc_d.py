# 2☆☆☆必ず解きなおす

# [考察]
# 派閥内の人は知り合いでなければならないので、積集合で求めようとしたけど
# n!通りの全探索になるので

# WA,TLE
n,m = map(int, input().split())
if m == 0:
    print(1)
    exit()
ans = {}
seen = set()
for i in range(m):
    x, y = map(int, input().split())
    if ans.get(x) == None:
        ans[x] = set()
    ans[x].add(y)
    ans[x].add(x)

    if ans.get(y) == None:
        ans[y] = set()
    ans[y].add(x)
    ans[y].add(y)

    seen.add(x)
    seen.add(y)

answer = 0
num = 0
seen = list(seen)

while num != 1:
    num = len(seen)
    for i in range(num-1):
        answer = max(answer, len(ans[seen[i]] & ans[seen[i+1]]))

    seen = seen[1:]
print(answer)

# ACコード
import itertools

n,m = map(int, input().split())
friend = [[0] * n for i in range(n)]
for i in range(m):
    x,y = map(int, input().split())
    x -= 1
    y -= 1
    friend[x][y] = 1
    friend[y][x] = 1

ans = 0

for bit in range(1 << n):
    group = []

    for i in range(n):
        if bit & (1 << i):
            group.append(i)

    flag = True

    for i in itertools.combinations(group, 2):
        if friend[i[0]][i[1]] == 0:
            flag = False
            break

    if flag:
        ans = max(ans, len(group))

print(ans)

# 84
# [考察]
# エラトステネスの篩かけて最初に10**5 以下の素数リストを用意
# その後累積和をやって2017-like-numberなら+1
# 最後にl,rを受け取ってO(1)で解答していく
# first - impressionの通りの問題であったが、計算量に信頼を置けず、
# 実装もしないまま終わってしまった
# 最悪である

q = int(input())
n = 10**5
ansli = [0]*(n+1)

def primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, n + 1):
        for j in range(i * 2, n + 1, i):
            is_prime[j] = False
    return is_prime

li = primes(n+1)
for i in range(3,n + 1):
    if li[i] and li[(i+1)//2]:
        ansli[i] = ansli[i - 1] + 1
    else:
        ansli[i] = ansli[i - 1]

for i in range(q):
    l, r = map(int, input().split())
    print(ansli[r] - ansli[l-1])

# 88.
# [考察]
# 全体のマス目から最短距離分のマス目だけが白であとは黒になればいいので、
# 最初の黒の数を記憶しておいて、幅優先探索で最短距離を求めて引く

from collections import deque

h, w = map(int, input().split())
s = [list(input()) for i in range(h)]
count = 0
for i in s:
    for j in range(w):
        if i[j] == "#":
            count += 1

def bfs():
    
    d = [[float("inf")] * w for i in range(h)]
    
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    que = deque([])
    que.append((0, 0))
    d[0][0] = 0

    while que:

        p = que.popleft()

        if p[0] == h - 1 and p[1] == w - 1:
            break

        for i in range(4):
            nx = p[0] + dx[i]
            ny = p[1] + dy[i]

            if 0 <= nx < h and 0 <= ny < w and s[nx][ny] != "#" and d[nx][ny] == float("inf"):
                
                que.append((nx, ny))
                d[nx][ny] = d[p[0]][p[1]] + 1

    return d[h - 1][w - 1]

ans = bfs()
if ans == float("inf"):
    print(-1)
else:
    print((h * w) - (ans + 1) - count)


# 94
# [考察]
# まずnは最大のものを選ぶ
# nCrでrを先に決めた時、(n+1)Cr > nCrだから
# rはnを除いてn//2 に近いものが正解
# (n - r) / (n + 1) でrがn // 2 を分かれ目として対称になるから
# combinationは必要ない

n = int(input())
a = list(map(int, input().split()))
x = max(a)
r = 0
num = x//2
diff = float("inf")
for i in range(n):
    dif = abs(num - a[i])
    if dif < diff and a[i] != x:
        diff = dif
        r = a[i]
print(x, r)

# 126☆☆☆
# [考察]
# 根からの距離が偶数のところは同じ色
# すべての頂点を同じ色でも良い
# 距離が偶数のところは同じ色
import sys
sys.setrecursionlimit(10**7)

n = int(input())
# 隣接リスト
link = [[] for i in range(n)]
for i in range(n-1):
    u, v, w = map(int, input().split())
    u -= 1
    v -= 1
    link[u].append((v,w))
    link[v].append((u,w))

ans = [-1]*n

def dfs(v, now):
    ans[v] = now

    for next_v ,w in link[v]:
        if ans[next_v] != -1:
            continue
        if w % 2:
            dfs(next_v, 1-now)
        else:
            dfs(next_v, now)

dfs(0,0)
for i in ans:
    print(i)

# 127
# [考察]
# 単純に元のリストに追加してソートして後ろからの合計とれば行けると思った
# けどいけなかった
# 結局最初の直感の考察が当たっており、一枚のカードにつき一回の書き換え、
# 書き換える数は大きいほうから、書き換えられるカードは小さいほうから
# 見ていくことでn回の操作だけで済むようになっている

# TLE
n, m = map(int, input().split())
a = list(map(int, input().split()))
for i in range(m):
    b, c = map(int, input().split())
    a += [c] * b
a.sort()
print(sum(a[-n:]))

# AC　計算量の見積もりが分からない

n, m = map(int, input().split())
a = sorted(list(map(int, input().split())))
li = []
for i in range(m):
    b, c = map(int, input().split())
    li.append((c, b))
li = sorted(li, reverse = True)

count = 0 #li[num][1]
num = 0  #index

for i in range(n):
    if a[i] < li[num][0]:
        a[i] = li[num][0]
        count += 1
    if count == li[num][1] :
        num += 1
        count = 0
    if num == m:
        break

print(sum(a))

# 130
# [考察]
# s(l,r)=∑[r,l]A[k]としたとき、s(a,b+1)>s(a,b),s(a,b)>s(a+1,b)より、
# s(l,r)>=kならば全てのx(x>=r)に対してもs(l,x)>=kが成立する
# そのような最初のrを見つける
# 実装は二分探索と尺取り法の二つがある

# TLE
# 累積和をとることにより連続部分文字列の総和をO(1)で得ることに成功したのに
# そのやり方がO(N**2)のためtime limit exceed

n,k = map(int, input().split())
a = list(map(int, input().split()))
li = [0]*(n+1)
count = 0

for i in range(1, n+1):
    li[i] = li[i-1] + a[i-1]

for i in range(1, k+1):
    for j in range(i, n+1):
        if li[j] - li[j-i] >= k:
            count += 1
print(count)

# 二分探索による解法
# 累積和から二分探索で最初にk以上になるところを探す
# 累積和のリストからいちいち引くことはできないので
# kに足すという工夫をした
# len(b),b = li[i:]の部分を工夫することでACさせた

import bisect

n,k = map(int, input().split())
a = list(map(int, input().split()))
ans = 0
li = [a[0]] + [0] * (n - 1)

for i in range(n-1):
    li[i + 1] = li[i] + a[i + 1]

if li[-1] < k:
    print(0)
    exit()

li = [0] + li[:]

for i in range(1, n + 1):
    num = bisect.bisect_left(li, k + li[i - 1])
    ans += (n+1) - num

print(ans)

# 尺取り法による解法

n, k = map(int, input().split())
a = list(map(int, input().split()))
ans = 0
right = 0
count = 0
for left in range(n):
    while (right < n and count < k):
        count += a[right]
        right += 1
    if count < k:
        break
    ans += n - (right - 1)
    count -= a[left]
    
print(ans)

# 131
# [考察]
# 締め切りが早いものから、そして始めなくては間に合わなくなる時間
# が早いものから選んでいき、始めなくてはならない時間よりも過ぎていたら失格

n = int(input())
li = []
work = []
for i in range(n):
    a,b = map(int, input().split())
    li.append((b,b-a,i))
    work.append(a)
li.sort()
now = 0
for bb,s,w in li:
    if now > s:
        print("No")
        exit()
    else:
        now += work[w]
print("Yes")

# 136
# [考察]
# 子供たちは必ず右か左に動くので、隣の子供と同じになることはない
# そして、子供たちは必ずRLという文字列に集まっている
# (RRRLLLという集合で考えた時)
# また、10**100は偶数であることから、その集合の偶数番目はR,Lの偶数番目のほう、
# 奇数番目は奇数番目の方に集まる
# 下の実装はsnukeさんのものを翻訳したやつ、Rについてだけ考えるように作られており、
# いったんRを見終わったら、文字列とリストをreverseして、R=>L,L=>Rという操作を行う
# ことで、Rについてだけ考える操作を可能にしている、もちろん最後に戻す
# このテクニックはどこかで使えるので頭においておく

s = list(input())
n = len(s)
ans = [0]*n
for _ in range(2):
    count = 0
    for i in range(n):
        if s[i] == "R":
            count += 1
        else:
            ans[i] += count//2 #L
            ans[i-1] += -(-count//2)#R
            count = 0
    ans = ans[::-1]
    s = s[::-1]
    for i in range(n):
        if s[i] == "L":
            s[i] = "R"
        else:
            s[i] = "L"

print(" ".join(map(str, ans)))

# 137
# コンテスト中に考えたやつ
# priority queというものを使うらしい
# WA
n,m = map(int, input().split())
li = [list(map(int, input().split())) for i in range(n)]
li.sort(key = lambda x:x[1],reverse = True)
day = 1
ans = 0
i = 0
while True:

    if day + li[i][0]  > m:
        break
    ans += li[i][1]
    day += 1
    i += 1
print(ans)

# Pythonだとheap_queueというデータ構造
# priority_queueは最大を取り出すけどheap_queueは最小を取り出すので-をつけていれる
# そしてまだ早い
n,m = map(int, input().split())
d = {}
for i in range(n):
    a,b = map(int, input().split())
    d.setdefault(a,[]).append(b)

# 138
# [考察]
# 数直線で考えてDFSして累積和

# コンテスト中に考えたものを終わった後に完成させた
# 木構造を自作するとこうなる
# 1WA 7TLE 6AC

from collections import defaultdict

n,q = map(int, input().split())
ans = dict.fromkeys(list(range(1,n+1)),0)

li = {i:i for i in range(1,n+1)}
li = defaultdict(set)

for j in range(n-1):
    a,b = map(int, input().split())
    li[a].add(b)

for k in range(q):
    p,x = map(int, input().split())

    if p == 1:
        for m in range(1,n+1):
            ans[m] += x
        continue

    if p in li:
        ans[p] += x
        for i in li[p]:
            ans[i] += x

    else:
        ans[p] += x

ans = list(ans.values())
print(" ".join(map(str,ans)))

# 139
# [考察]
# 余りが最大になるのはn-1で割ったとき

n = int(input())
print((n*(n-1))//2)

# 141
# [考察]
# その状態で最も値段の高いものに使っていくので、愚直に計算すると
# a[i] = max(a)みたいな感じでやっていき、O(mn)になって間に合わない
# ここで、priority_queue(heapq)を使う
# pythonのheapqは最小値を取り出すので*(-1)して最大値取り出しに対応させる

import heapq
n,m = map(int, input().split())
a = list(map(lambda x: int(x)*(-1), input().split()))
heapq.heapify(a)

for i in range(m):
	target = heapq.heappop(a)
	heapq.heappush(a, (-1)*(-target//2))

print(-sum(a))


# 142
# TLE & WA

a,b = map(int, input().split())

def make_divisors(n):
    divisors = set()
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divisors.add(i)
            if i != n // i:
                divisors.add(n//i)

    # divisors.sort()
    return divisors
div = make_divisors(min(a,b))
ali = make_divisors(a)
bli = make_divisors(b)
ansli = ali & bli
print(ansli)

n = max(ansli)
primes = {2}
for i in range(3, n, 2):
    all(i % p != 0 for p in primes) and primes.add(i)
ans = primes & ansli
print(len(ans) + 1)

# AC
# [考察]
# 最大公約数の素因数+1(1の分)

a,b = map(int, input().split())

def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a%b)

def prime_factorize(n):
    a = []
    while n % 2 == 0:
        a.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            a.append(f)
            n //= f
        else:
            f += 2
    if n != 1:
        a.append(n)
    return a

print(len(set(prime_factorize(gcd(a,b))))+1)
