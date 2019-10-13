[遊び]
任意の自然数の各桁を一けたになるまで掛け算する回数の最大回数とその数を示す
問題の、掛け算の回数を求めるプログラム
n = input()
count = 0
while len(n) != 1:
    li = list(n)
    li = [int(i) for i in li]
    print(li)
    ans = 1
    for j in range(len(li)):
        ans *= li[j]
    n = str(ans)
    count += 1
print(count)

42
[考察]
n以上で愚直に増やしていく

n,k = map(int, input().split())
d = set(list(map(str, input().split())))
cost = n
while True:
    li = list(str(cost))
    if all(i not in d for i in li):
        break
    else:
        cost += 1
print(cost)

43
[考察]
計算量的にa[0]-a[-1]の範囲で全部やっても間に合うので調べる

n = int(input())
a = sorted(list(map(int, input().split())))
ans = float("inf")

for i in range(a[0], a[-1]+1):
    count = 0
    for j in range(n):
        count += (a[j] - i)**2
    ans = min(ans, count)
print(ans)

45 bit全探索
[考察]
n-1通りでプラスを入れるかどうか全探索したいのでbit全探索をする
[注意]
ビット全探索を考えるときはビットの右も左も考える必要がない!!!

s = input()
n = len(s)
ans = 0

for bit in range(1 << n - 1):
    f = s[0]

    for i in range(n - 1):
        if bit & (1 << i):
            f += "+"
        f += s[i + 1]

    ans += sum(map(int, f.split("+")))

print(ans)

WA
s = input()
n = len(s)
ans = 0

import copy

for bit in range(1 << n-1):
    target = copy.copy(s)

    for i in range(n-1):
        if bit & (1 << i):

            ans += int(target[-i-1:])
            target = target[:-i-1]

    ans += int(target)
print(ans)

47
[考察]
実験により得た。右隣が違った回数だけやる
bbb,wwといった同じ文字の連続は一つの集合と捉えて
b,wだけで置き換えられるので、結局区間の数が答えとなる

s = input()
ans = 0
for i in range(len(s)-1):
    if s[i] != s[i+1]:
        ans += 1
print(ans)

48
[考察]
できるだけ右から減らしていく
右は次のために更新するけど左はもう使わないので更新しない

n,x = map(int, input().split())
a = list(map(int, input().split()))
ans = 0
for i in range(n-1):
    num = a[i] + a[i+1]
    if num > x:
        ans += num - x
        if a[i+1] - (num - x) < 0:
            a[i+1] = 0
        else:
            a[i+1] -= num - x
print(ans)

49
[考察]
消していく順番に注意する

s = input()
s = s.replace("eraser","").replace("erase","").replace("dreamer","").replace("dream","")
if s:
    print("NO")
else:
    print("YES")

50
[考察]
nが偶数
絶対値は全て奇数で最大はn-1、すべての奇数が2回出てくる、答えは2**(n//2)
nが奇数
絶対値は全て偶数で最大はn-1、0は1回、その他は2回出てくる、答えは2**((n-1)//2)

絶対値がすべて奇数もしくは偶数でちゃんと2ずつ上がってて、最大がn-1であるという条件
を満たしているかを、xorを使うことで確認しました
xor(対称差集合)を使ってlen()が0出なかったら条件に合っていない数が含まれている
ということになります

import collections

n = int(input())
a = list(map(int, input().split()))
li = collections.Counter(a)
key = set(li.keys())
flag = True
mod = (10**9)+7

if n % 2:
    num = set(list(range(0,n,2)))#n-1
    if len(key ^ num) != 0:
        flag = False

    for i in li:
        if i == 0 and li[i] != 1:
            flag = False

        elif i != 0 and li[i] != 2:
            flag = False

    if flag:
        print((2**((n-1)//2))%mod)
    else:
        print(0)
else:
    num = set(list(range(1,n,2)))
    if len(key ^ num) != 0:
        flag = False

    for i in li:
        if li[i] != 2:
            flag = False
    if flag:
        print((2**(n//2))%mod)
    else:
        print(0)

51
[考察]
サンプルと同じように通ればよい
必ず右上にあるので考える必要もない

sx, sy, tx, ty = map(int, input().split())

length = ty - sy
width = tx - sx
u = "U"
l = "L"
r = "R"
d = "D"
ans = ''
ans += r*width + u*length
ans += l*width + d*length
ans += d + r*(width+1) + u*(length+1) + l
ans += u + l*(width+1) + d*(length+1) + r
print(ans)

52
[考察]
n!で一回一回素因数分解してどの素因数が何回で出てきたか数える
最後にそれぞれ1足してかけ合わせてmod
import collections

n = int(input())

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

li = {}

for i in range(2, n+1):

    c = collections.Counter(prime_factorize(i))
    keys = list(c.keys())
    values = list(c.values())
    for i in range(len(keys)):
        if keys[i] in li:
            li[keys[i]] += values[i]
        else:
            li[keys[i]] = values[i]

ans = list(li.values())
ans = [i+1 for i in ans]

count = 1
mod = (10**9) + 7
for i in ans:
    count = count * i % mod
print(count)

53
[考察]
6以下なら一回目で6出して終わり、それ以外は6-5を繰り返すので
11で割ったあまりで考える
余りが6以上なら11-6-5までやらないといけないので+2
余りが0なら11の倍数なので商
6以下なら11-6で終わるので+1

x = int(input())
if x <= 6:
    print(1)
else:
    num = x % 11
    if num == 0:
        print((x//11)*2)
    elif num > 6:
        print((x//11)*2 + 2)
    else:
        print((x//11)*2 + 1)

54 ☆☆
[考察]
dfs
制約が<=8のときはn!が間に合う
ので全探索する
n-1回の巡回ですべて巡れていたならカウント
典型の中でも単純な問題

n,m = map(int, input().split())
link = [[] for i in range(n)]
for i in range(m):
    a,b = map(int, input().split())
    a -= 1
    b -= 1
    link[a].append(b)
    link[b].append(a)

visited = [0]*n
ans = 0

def dfs(now, prev):
    global ans

    if sum(visited) == n:
        ans += 1

    for i in link[now]:
        if i != now and visited[i] == 0:
            visited[i] = 1
            dfs(i, now)
            visited[i] = 0

visited[0] = 1
dfs(0,-1)
print(ans)

55
[考察]
o cc => s, × cc <= s

n,m = map(int, input().split())
if n > m//2:
    print(m//2)
else:
    print((n+(m//2))//2)

56
[考察]
xというのは等差1の数列の和がx以上になったとき、
必ずどれか一つ(1,2..n)を引くことで表現できる

x = int(input())
i = 0
num = 0
while True:
    num += i
    if num >= x:
        break
    i += 1
print(i)

57
[考察]
約数をO(n**(1/2))で列挙してそれぞれに対して
大きいほうの桁数が最小になるようにする
約数がN**(1/2)のときはリストに同じ数を追加して
リストの長さを偶数にする

n = int(input())
def make_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n//i)
    return divisors

ans = float("inf")
li = make_divisors(n)
if li[-1]**2 == n:
    li.append(li[-1])

for i in range(len(li))[::2]:
    ans = min(ans, max(len(str(li[i])), len(str(li[i+1]))))
print(ans)

58
[考察]
汚いコード。セットで共通するもの出して共通する文字の少ないほうを辞書
に記憶する

import collections
import copy

n = int(input())
now = list(input())
ans = collections.Counter(now)
now = set(now)

for i in range(n-1):
    s = list(input())
    li = collections.Counter(s)
    now = set(s) & now
    d = ans.copy()
    ans = dict()
    for i in now:
        ans[i] = min(d[i], li[i])

if len(list(ans.keys())) == 0:
    print("")
    exit()
ansli = []
for i in ans:
    ansli += [i]*ans[i]

ansli.sort()
print("".join(map(str, ansli)))

60
[考察]
間の時間が流れる時間より長かったら最後まで流す
そうでないなら間の時間ながす

n,T = map(int, input().split())
t = list(map(int, input().split()))
ans = T
for i in range(n-1):
    diff = t[i+1] - t[i]
    if diff >= T:
        ans += T
    else:
        ans += diff
print(ans)

61
[考察]
方法がごり押ししかないので辞書で頑張る

n, k = map(int, input().split())
li = dict()

for i in range(n):
    a,b = map(int, input().split())
    li.setdefault(a,0)
    li[a] += b

li = sorted(li.items())
ans = 0
for i in range(len(li)):
    ans += li[i][1]
    if ans >= k:
        print(li[i][0])
        exit()

63
[考察]
target != inf and ans % 10 == 0
という状況はすべての10の倍数でない数がうまくペアを作って
10の倍数になっているということなので、10の倍数でない最小の値をひいてやれば
ペアが崩れて全体として10の倍数でない数になる

n = int(input())
target = float('inf')
ans = 0

for i in range(n):
    s = int(input())
    ans += s
    if str(s)[-1] != "0":
        target = min(target, s)

if ans % 10 != 0:
    print(ans)
elif ans % 10 == 0 and target == float('inf'):
    print(0)
else:
    print(ans - target)

cf
n = int(input())
s = sorted([int(input()) for _ in range(n)])
ans = sum(s)
if ans % 10 != 0:
    print(ans)
else:
    for i in s:
        if i % 10 != 0:
            print(ans - i)
            exit()
    print(0)


64
[考察]
最大は3199以下の種類数 + 3200以上の人数
最小はもし3199以下が0なら==3200以上だけなら全員が同じ色にして1
それ以外は3199以下の種類数のみ

n = int(input())
a = list(map(int, input().split()))
li = [0]*15
tourist = 0

for i in a:
    if i // 400 >= 8:
        tourist += 1

    if li[i//400] == 0:
        li[i//400] = 1

maxrate = sum(li[:8]) + tourist

minrate = 0

if sum(li[:8]) == 0:
    minrate = 1
else:
    minrate = sum(li[:8])

print(minrate, maxrate)

65
[考察]
隣り合わないようにするにはまず差が1以下でなければならない
階乗はmodをとりながらfor文ループでやるのがよい
差が1なら、最初のやつは多いほうに決定されるので一通りしかない
よって下のように掛け算しながらmodとってやる
差が0ならどちらから始めてもいいので2通りあるので下のように

n,m = map(int, input().split())
mod = (10**9) + 7

if abs(n-m) > 1:
    print(0)
    exit()

li = sorted([n,m])

num = 1
for i in range(1,li[0]+1):
    num = (num * i) % mod

if li[1] - li[0] == 1:
    print((((num * li[1]) % mod) * num) % mod)
else:
    print((((num * num) % mod) * 2) % mod)

66
[考察]
nが偶数だったら奇数番目が真ん中より左に逆順、偶数番目が右にそのまま
奇数なら奇数番目が右にそのまま、偶数番目が左に逆順
最後に全体を逆順にする

n = int(input())
a = list(map(int, input().split()))
if n % 2:
    li = a[1::2][::-1] + a[::2]
    li = li[::-1]
else:
    li = a[::2][::-1] + a[1::2]
    li = li[::-1]
print(" ".join(map(str, li)))

67
[考察]
総和とそこまでの和との差を最小化する

n = int(input())
a = list(map(int, input().split()))
now = 0
ans = float("inf")
total = sum(a)
for i in range(n-1):
    now += a[i]
    x = now
    y = total - now
    ans = min(ans, abs(x-y))

print(ans)

68
[考察]
スタートからどこか、どこかからゴールという行き方しかないので
スタートからの止まる場所、ゴールに向かう場所をそれぞれ出して、
一致するものが一つでもあれば可能、セットの積集合を使う

n,m = map(int, input().split())
start = set()
goal = set()
for i in range(m):
    a,b = map(int, input().split())
    if a == 1:
        start.add(b)
    elif  b == n:
        goal.add(a)

ans = start & goal
if len(ans) > 0:
    print("POSSIBLE")
else:
    print("IMPOSSIBLE")

69
[考察]
問題文をしっかり読む、a[i] * a[i+1]なので
奇数が4の倍数より少なかったら141414の形を作って後は適当にできる
多かったら14141というfour + 1 == oddの個数だったらできる
そして右端の1の次に4k以外の偶数があったらダメなので0個でなくてはならない

n = int(input())
a = list(map(int, input().split()))
odd = 0
even = 0
four = 0

for i in a:
    if i % 4 == 0:
        four += 1
    elif i % 2 == 0:
        even += 1
    elif i % 2:
        odd += 1

if odd > four:
    if not (odd -1 == four and even == 0):
        print("No")
        exit()
print("Yes")

70
[考察]
3つ以上の最小公倍数の出し方

def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a%b)

def lcm(x,y):
    return (x * y) // gcd(x,y)

n = int(input())
now = int(input())
for i in range(n-1):
    t = int(input())
    now = lcm(now,t)
print(now)

71
[考察]
4本以上あるなら正方形を作ってみる(最大の一辺)
2本以上が2つ以上あるならその中でも最も長い二つで長方形を作る
それらを比較する
正方形もできない、長方形も作れないならダメ


import collections

n = int(input())
a = list(map(int, input().split()))
li = collections.Counter(a)
square = 0
rectangle = []
for i in li:
    if li[i] >= 2:
        rectangle.append(i)
    if li[i] >= 4:
        square = max(square, i)

rectangle.sort()

if len(rectangle) < 2 and square == 0:
    print(0)
elif len(rectangle) < 2:
    print(square**2)
else:
    print(max(square**2, rectangle[-1] * rectangle[-2]))

72
[考察]
連続する3つの整数の数の合計が最も多いものが答え

import collections

n = int(input())
a = list(map(int, input().split()))
li = collections.Counter(a)
ans = 0
for i in li:
    ans = max(ans, li[i]+li[i+1]+li[i+2])
print(ans)

73
[考察]
辞書を使って紙に書かれている状態を1、消してある状態を0として
最終的な1の個数を出力

n = int(input())
li = dict()
for i in range(n):
    a = int(input())
    li.setdefault(a,0)
    if li[a] == 0:
        li[a] = 1
    else:
        li[a] = 0
ans = list(li.values())
print(ans.count(1))

76
[考察]
すでにtが入っていたとき、?があればaで埋めて出力
|t|だけ切り取って、?なら必ず入れられるから、?ではないときを考える
?でないかつi番目がsと一致しているなら続ける
もし一つでも一致していなかったら入れられないので
スルー、その区間にtを入れることができると判断したら置き換えてとっておく
この時、replace()を使うとすべてやってしまうので、文字列の連結を使う
最後に、ソートして一番最初の文字列について、まだ?があればaで埋める
なければそのまま出力

s = input()
t = input()
li = []

if t in s:
    if "?" in s:
        print(s.replace("?", "a"))
        exit()
    else:
        print(s)
        exit()

for i in range(len(s) - len(t) + 1):
    target = s[i:i+len(t)]
    flag = True
    for j in range(len(t)):
        if target[j] != "?" and target[j] != t[j]:
            flag = False
    if flag:
        li.append(s[:i] + t + s[i+len(t):])

if len(li) == 0:
    print("UNRESTORABLE")
    exit()

li = sorted(li)
print(li[0].replace("?", "a"))

78
[考察]
1回で1900msの答えがm回あって全体で(2**m)回繰り返される
さらに100msの答えは一回で(n-m)回あって全体で(100*n-m*(2**m))
これらを合計して答え


n,m = map(int, input().split())
num = 2**m
print((1900*m*num) + ((n-m)*100*num))

79 bit全探索
自分のコード

li = input()
for bit in range(1 << 3):
    f = li[0]

    for i in range(3):
        if bit & (1 << i):
            f += "+"
        else:
            f += "-"
        f += li[i+1]

        if i == 2:
            ans = int(f[0])
            for i in range(6):
                if f[i] == "+":
                    ans += int(f[i+1])
                elif f[i] == "-":
                    ans -= int(f[i+1])
            if ans == 7:
                print(f + "=7")
                exit()

cf
[考察]
fは出力用に作ってansで見るのでif-elseのところで計算してしまう
s = input()

for bit in range(1 << 3):
    ans = int(s[0])
    f = s[0]

    for i in range(3):
        # フラグが立っていれば "+" 、なければ "-"
        if bit & (1 << i):
            ans += int(s[i + 1])
            f += "+"
        else:
            ans -= int(s[i + 1])
            f += "-"
        f += s[i + 1]

    if ans == 7:
        print(f + "=7")
        exit()

[再帰]
ベースケース=終了条件を一番上に書く
その下に終了条件に近づくように書いてあとはいろいろ計算式を書く
def dfs(i, f, sum):
    if i == 3:
        if sum == 7:
            # 答えは1つ出力すれば良いので =7 になれば終了
            print(f + "=7")
            exit()

    else:
        # 式 f の末尾に符号と次の数字を追加し、その分 sum に加減する
        dfs(i + 1, f + "-" + s[i + 1], sum - int(s[i + 1]))
        dfs(i + 1, f + "+" + s[i + 1], sum + int(s[i + 1]))


s = input()

dfs(0, s[0], int(s[0]))

81
[考察]
k種類以下になるまで最も個数の少ない数を書き換えていく

import collections
n,k = map(int, input().split())
a = list(map(int, input().split()))
c = collections.Counter(a)
li = c.most_common()
count = m = len(li)
ans = 0
for i in range(1,m+1):
    if count <= k:
        break
    count -= 1
    ans += li[-i][1]
print(ans)

82
[考察]
xがx個より少ない=>n個取り除く
多い=>n-x個取り除く

import collections

n = int(input())
a = list(map(int, input().split()))
li = collections.Counter(a)

ans = 0
for i in li:
    if li[i] < i:
        ans += li[i]
    elif li[i] > i:
        ans += li[i] - i
print(ans)

83
[考察]
範囲内でひたすら2をかける

x,y = map(int , input().split())
count = 0
n = x
while n <= y:
    n *= 2
    count += 1
print(count)

84
[考察]
前の電車の発車時間に次の駅までにかかる時間を加えたものを今の時間として
次の電車の発車時間と比較する
その駅の最初の発車時間がまだ来てなかったら最初の時間に乗る
もう来てたら次の発車時間を調べたいのでs%f==0の性質を利用して
到着した時間とその駅の次の発車時間との間の時間を割ってかけたものが
次の電車の発車時間になり、それはfの倍数である

85 全探索 2重のfor文
n,y = map(int, input().split())
for a in range(n+1):
    for b in range(n+1):
        c = n - a - b
        if 10000*a + 5000*b + 1000*c == y and 0 <= c <= n :
            print(a, b, c)
            exit()
print("-1 -1 -1")

86
[考察]
移動距離が時間の差よりも大きいなら無理、小さくても偶奇が一致していないければ
そこにはたどり着けないのでこれらを判定する

n = int(input())
li = [list(map(int, input().split())) for i in range(n)]
li = [[0,0,0]] + li

for i in range(n):
    t_diff = li[i+1][0] - li[i][0]
    xy_diff = sum(li[i+1][1:]) - sum(li[i][1:])

    if t_diff < xy_diff or t_diff % 2 != xy_diff % 2:
        print("No")
        exit()
print("Yes")

87
[考察]
一回下に行ったら上には戻れないので、どこから下に行くかを問題にする
そのため最初に上の累積和をとっておき、左から行った場合との大きいほうを
取り続ける

n = int(input())
a = [list(map(int, input().split())) for i in range(2)]

li = [[a[0][0]] + [0]*(n-1), [a[0][0]+a[1][0]] + [0]*(n-1)]
for i in range(1,n):
    li[0][i] = li[0][i-1] + a[0][i]

for i in range(1,n):
    li[1][i] = max(li[0][i] + a[1][i], li[1][i-1] + a[1][i])

print(li[1][n-1])

cf 考察を端的に表現したもの
n = int(input())
a1 = list(map(int, input().split()))
a2 = list(map(int, input().split()))

ans = 0
for i in range(n):
    ans = max(ans, sum(a1[:i+1]) + sum(a2[i:]))
print(ans)


88
[考察]
最初にa1,b1を固定して、その状態からa2,a3,b2,b3は一つに決まる
決定された6つでc[i][j]に対してうまくいくなら〇
すべてだめなら×

c = [list(map(int, input().split())) for i in range(3)]

for i in range(c[0][0]+1):

    a = [0]*3
    b = [0]*3
    a[0] = i
    b[0] = c[0][0] - i

    for j in range(1,3):
        a[j] = c[j][0] - b[0]
        b[j] = c[0][j] - a[0]
    flag = True
    for k in range(3):
        for l in range(3):
            if c[k][l] != a[k] + b[l]:
                flag = False
    if flag:
        print("Yes")
        exit()

print("No")

89
[考察]
文字の組み合わせは最大でも10通りしかないので、出てきた文字列たちの
組み合わせを列挙して、それぞれに対してその文字の個数を
かけたものを足していく
setdefault()と辞書の更新とのコンボ

n = int(input())
li = set(list("MARCH"))

name_li = set()
name = dict()

for i in range(n):
    s = input()
    if s[0] in li:
        name.setdefault(s[0], 0)
        name[s[0]] += 1
        name_li.add(s[0])

import itertools
c_list = list(itertools.combinations(name_li, 3))

ans = 0

for i in c_list:
    ans += name[i[0]] * name[i[1]] * name[i[2]]
print(ans)

cf
from collections import Counter
import itertools
N = int(input())
S = [input()[0] for _ in range(N)]

counter = Counter(S)
answer = 0
for x,y,z in itertools.combinations('MARCH',3):
  answer += counter[x] * counter[y] * counter[z]

print(answer)


90
[考察]
縁にあるものは接する数が奇数になるので必ず表になる
よって全体の数から四縁の数を引く
絶対値をとれば行数が1の時などにも対応できる

n,m = map(int, input().split())
print(abs((n*m)-(n*2)-(m-2)*2))

言い換えれば(n-2)*(m-2)の枚数

n,m = map(int, input().split())
print(abs((n-2)*(m-2)))

92
[考察]
汚いコード、まず各区間の距離をとる、次に一つ飛ばした距離をとる
そして各距離の総和から、その距離を飛ばすことを考えて区間1-2,2-3の距離を引いて
区間1-3の距離を足したものが答え
他の距離は変化しないので該当部分だけ処理する

n = int(input()) + 1
a = list(map(int, input().split()))
a = [0] + a
dist_one = []
skip = []
for i in range(n-1):
    dist_one.append(abs(a[i+1] - a[i]))

    if i == n-2:
        skip.append(abs(a[i] - a[0]))
    else:
        skip.append(abs(a[i+2] - a[i]))

dist_one.append(abs(a[-1] - a[0]))

base = sum(dist_one)

for i in range(n-1):
    print(base - sum(dist_one[i:i+2]) + skip[i])

93
[考察]
最大に対する距離で考える
距離の偶奇が異なっているときに注意

[解説]
3x-(a+b+c)/2
=> x(一致したときの数) >= max(a,b,c) と 3x === a+b+c (mod2)
となる最小のxを求める
xはm = max(a,b,c)もしくは m+1


a,b,c = map(int, input().split())
li = sorted([a,b,c])
ans = 0

if (li[2]-li[0])%2 != (li[2]-li[1])%2:

    if li[2]%2 == li[0]%2:
        li[2] += 1
        li[0] += 1
    else:
        li[2] += 1
        li[1] += 1
    ans += 1

d = min(li[2]-li[0],li[2]-li[1])
e = max(li[2]-li[0],li[2]-li[1])
ans += d + (e-d)//2
print(ans)

94
[考察]
削除する値が真ん中より前なら後ろから(n//2)番目が中央値
真ん中より後なら前から(n//2)-1番目が中央値
辞書でとるのがよい

n = int(input())
x = list(map(int, input().split()))
ans = dict()
x_copy = sorted(x[:])
right = x_copy[(n//2)-1]
left = x_copy[-(n//2)]

for i in range(n):
    if i < n//2:
        ans[x_copy[i]] = left
    else:
        ans[x_copy[i]] = right

for i in x:
    print(ans[i])

95
[考察]
適切な場合分け
①ABだけでいく、②ABで少ないほうまでいって多いほうの残りを単品、
③すべて単品

a,b,c,x,y = map(int, input().split())
one = max(x,y) * 2 * c
two = min(x,y) * 2 * c
if x >= y:
    two += abs(x-y) * a
else:
    two += abs(x-y) * b
three = x * a + y * b
print(min(one, two, three))

96
[考察]
hとwを間違えないように気を付けましょう
"#"の四方に"#"がいなかったら2つ黒くぬることができないのでだめです

h,w = map(int, input().split())
s = [list(input()) for i in range(h)]
dx = [0,1,0,-1]
dy = [1,0,-1,0]

for x in range(h):
    for y in range(w):
        if s[x][y] == "#":
            flag = False

            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]

                if 0 <= nx < h and 0 <= ny < w and s[nx][ny] == "#":
                    flag = True

            if not flag:
                print("No")
                exit()

print("Yes")

98
[考察]
O(n**2)だと間に合わないのでO(n)を考えたいとなったときに
累積和を使うことで実現させる
具体的には左のwの数+右のEの数の合計を最小化させたいので
最初にその時のEの数とWの数を持っておいてs[i]がリーダーになったとに
いくつなのかを高速で答える

n = int(input())
s = input()
ans = float("inf")
e = [0]*(n+1)
w = [0]*(n+1)
for i in range(n):
    if s[i] == "E":
        e[i+1] = e[i] + 1
        w[i+1] = w[i]
    else:
        e[i+1] = e[i]
        w[i+1] = w[i] + 1

for i in range(n+1):
    ans = min((e[-1]-e[i])+(w[i-1]), ans)

print(ans)

100
[考察]
全部奇数になったらダメ=>何回2で割れるか
なので2で割れる回数を足す

n = int(input())
a = list(map(int, input().split()))
ans = 0
for i in range(n):
    count = 0
    while a[i] % 2 == 0:
        count += 1
        a[i] /= 2
    ans += count
print(ans)

101☆
[考察]
グループ分けする問題、1の位置は関係ないので受け取らない
最初はk個なので最後に1を足す、毎回1かぶるのでk-1、
k-1でぴったり終わらないならさらに+1

n,k = map(int, input().split())
if (n-k)%(k-1) == 0:
    print((n-k)//(k-1)+1)
else:
    print((n-k)//(k-1)+2)

103
[考察]
余りは割る数-1が最大ですべての数に対してそれを満たすような数がある
という保証があるはず

n = int(input())
a = list(map(int, input().split()))
ans = 0
for i in range(n):
    ans += a[i]-1
print(ans)

104 ☆bit全探索
[テクニック]
min()をするときに比較対象をfloat("inf")にしておく
切り上げはmathではなく-(-4 // 3)のように書く
indexで記憶する,その時重複はないのでsetを使うと1000倍くらい早くなることもある


d,g = map(int, input().split())
pc = [list(map(int, input().split())) for i in range(d)]

ans = float("inf")

for bit in range(1 << d):
    total = 0
    count = 0
    left = set(range(1, d+1))

    for i in range(d):
        if bit & (1 << i):
            total += pc[i][0] * 100 * (i+1) + pc[i][1]
            count += pc[i][0]
            left.discard(i+1)
            #使ったものとして記憶

    if g > total:
        target = max(left)
        n = min(pc[target - 1][0], -(-(g - total) // (target * 100)))#切り上げ
        count += n
        total += n * target * 100
    #この下とはつながっていない

    if g <= total:
        ans = min(ans, count)

print(ans)

再帰
def dfs(i, total, count, left):
    global ans
    if i == d:
        if total < g:
            target = max(left)
            n = min(pc[target - 1][0], -(-(g - total) // (target * 100)))
            count += n
            total += n * target * 100

        if total >= g:
            ans = min(ans, count)

    else:
        dfs(i+1, total, count, left)
        dfs(i + 1, total + pc[i][0] * (i + 1) * 100 + pc[i][1], \
        count + pc[i][0], left - {i + 1})

d,g = map(int, input().split())
pc = [list(map(int, input().split())) for i in range(d)]

ans = float("inf")

dfs(0,0,0,set(range(1,d+1)))
print(ans)

106
[考察]
kが最初からの連続する1の数以下なら1
それ以外は1ではない最初の数字

s = input()
k = int(input())
count = 0
ans = 0
for i in range(len(s)):
    if s[i] == "1":
        count += 1
    else:
        ans = int(s[i])
        break
print(1 if count >= k else ans)

107
[考察]
入力の時点で昇順に並んでいる=>左端が負で右端が正なら、正負をまたいでいるので
往復して短いほうを選ぶ。どちらも負もしくは正なら、負なら左端まで行き正なら右端
まで行くのでどちらにも対応できるようにmax()とったところまで行く

n, k = map(int, input().split())
x = list(map(int, input().split()))
ans = float("inf")

for i in range(n-k+1):
	count = 0
	left = x[i]
	right = x[i+k-1]

	if left < 0 and right >= 0:
		a = abs(left)*2 + right
		b = right*2 + abs(left)
		count = min(a, b)
		ans = min(ans, count)
	else:
		count = max(abs(x[i]), abs(x[i+k-1]))
		ans = min(ans, count)

print(ans)

108
[考察]
まるで歯が立たなかった。mod kで考える
a+b,b+c,c+aがすべてkの倍数であるためにはa,b,cすべてのmod k が
0またはk/2でなければならない
kが奇数の時はk/2はないのでkの倍数**3が答え
偶数の時はkの倍数**3とk/2の倍数を考えるのだがk/2はkの倍数を含むことがある
のでn//(k//2) - (n//k)をして3乗する
[参考]
①(a+b+...)%k = 0のとき、
  a%k + b%k + ... = nk(n = 1,2,...)

②倍数を考える問題=>modを考える

n,k = map(int, input().split())
ans = 0
if k % 2:
    ans = (n//k)**3
else:
     even = n//k
     odd = (n//(k//2)) - even
     ans = even**3 + odd**3
print(ans)

109
[考察]
最初の座標も入れて最大公約数をとる

n,X = map(int, input().split())
x = list(map(int, input().split()))
if n == 1:
    print(abs(X-x[0]))
    exit()

def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a%b)

x = x + [X]
x.sort()
ans = x[1] - x[0]

for i in range(1,n):
    ans = gcd(ans, x[i+1] - x[i])
print(ans)

111
[考察]
基本的には最頻値を全体から引くことで求められる
最頻値を持つ種類が同じだったときは次に多い数の多いほうにかえる

from collections import Counter
n = int(input())
v = list(map(int, input().split()))
a = Counter(v[::2])
b = Counter(v[1::2])
count = a.most_common()
count1 = b.most_common()
if len(set(v)) == 1:
    print(n//2)
    exit()
if count[0][0] == count1[0][0]:
    print(n- count[0][1] - (max(count[1][1], count1[1][1])))

else:
    print(n - count[0][1] - count1[0][1])

113
[考察]
実装が天才だった、タプルで順番をとっておく
県と誕生順はリストで記憶しておく
出力の時にやるだけになるように
計算量が分からないのでこの違いもよくわかりません

AC
n,m = map(int, input().split())
py = []
for i in range(m):
    p,y = map(int, input().split())
    py.append((p,y,i))
py.sort()
li = [[0]*2 for i in range(m)]
s = 0
t = 0
for p,y,i in py:
    #もしs!=pならその県は初めてなので1番目、違うならそこから+1
    if s != p:
        s = p
        t = 1
    else:
        t += 1
    li[i][0] = s
    li[i][1] = t
for i in range(m):
    print(str(li[i][0]).zfill(6) + str(li[i][1]).zfill(6))

TLE
import numpy as np
n,m = map(int, input().split())
count = [0]*n
py = []

for i in range(m):
    p,y = map(int, input().split())
    count[p-1] += 1
    py.append((p,y,i))
count = np.cumsum(count)

py = sorted(py)
print(py)

for i in range(m):
    num = li.index(py[i]) + 1
    pref = str(py[i][0])
    if py[i][0] == 1:
        print(pref.zfill(6) + str(num).zfill(6))
    else:
        print(pref.zfill(6) + str(num - count[i]).zfill(6))

114
[考察]
3,5,7のどれかで構成される文字==準七五三数を全列挙
再帰呼び出ししてその数が七五三数だったら+1

n = int(input())

def dfs(s):
    if int(s) > n:
        return 0
    if all(s.count(c) > 0 for c in "753"):
        ret = 1
    else:
        ret = 0


    for c in "753":
        ret += dfs(s + c)

    return ret

print(dfs("0"))

115
[考察]
ソートしてi+k-1本目との差が最も小さいものを答えとする
無駄に差のリストとか出して計算量を増やさないこと

n, k = map(int, input().split())
h = sorted([int(input()) for i in range(n)])
ans = float("inf")

for i in range(n-k+1):
    ans = min(ans, h[i+k-1]-h[i])
print(ans)

116☆☆☆
[考察]
これは難しい、頂点数から辺の数を引くと求める数になるらしいんですが、
保留です

n = int(input())
h = list(map(int, input().split()))
h.insert(0, 0)
ans = 0

for i in range(n):
    ans += max(h[i + 1] - h[i], 0)
print(ans)

117☆
[考察]
n-1個の仕切りを入れる、区切りごとのコストを最小化する
difficultyよりも難しい
仕切りを入れる=>間の距離をひとつ無視することができる
よって間の距離が大きいものから捨てていくという操作

n, m = map(int, input().split())
x = sorted(list(map(int, input().split())))
dist = sorted([x[i+1] - x[i] for i in range(m-1)], reverse = True)
print(sum(dist[n-1:]))

118☆
[考察]
複数の最大公約数をユークリッドの互除法を使って求める
このような問題に早く気付けるように

n = int(input())
a = list(map(int, input().split()))

def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a%b)

ans = a[0]
for i in range(1,n):
    ans = gcd(ans, a[i])
print(ans)

119

120
[考察]
0と1が一つ以上残っていたら必ずどこかで消せるのでそれぞれ数えて
少ないほうの数だけペアが作れるので×2して計算する

s = input()
count0 = s.count("0")
count1 = s.count("1")
print(2 * min(count0, count1))

121
[考察]
安いものから単純に買っていくので最初にソートするだけ

n, m = map(int, input().split())
ab = sorted([list(map(int, input().split())) for i in range(n)])
count = 0
ans = 0

for i in range(n):

    if count + ab[i][1] >= m:
        ans += ab[i][0] * (m - count)
        print(ans)
        exit()

    else:
        count += ab[i][1]
        ans += ab[i][0] * ab[i][1]

122
[考察]
事前に記憶しておくという意味では同じだったけど範囲を与えられたときに
すぐに答えられるように用意しておく必要があるので、その位置までの数を記憶しておいて
累積和のような使い方をする

n,q = map(int, input().split())
s = input()
li = [0]*(n+1)
for i in range(n):
    if s[i:i+2] == "AC":
        li[i+1] = li[i] + 1
    else:
        li[i+1] = li[i]

for i in range(q):
    l, r = map(int, input().split())
    print(li[r-1] - li[l-1])

<TLE>
n,q = map(int, input().split())
s = input()
li = []
for i in range(n-1):
    if s[i] == "A" and s[i+1] == "C":
        li.append(i+1)
for i in range(q):
    l, r = map(int, input().split())
    ans = 0
    for num in li:
        if l <= num and num+1 <= r:
            ans += 1
    print(ans)

123
[考察]
i-1分後から乗る,ゴールから5-i分前には使い終わる
(i-1) + (5-i) = 4分間は誰も乗っていない
切り上げ = -(-num // n)

n = int(input())
num = float('inf')
for i in range(5):
    m = int(input())
    num = min(num, m)
print(-(-n // num) + 4)

124
[考察]
制約を見ればループを二回回せることが分かるので
0スタートと1スタートで少ないほうをとる

s = input()
n = len(s)
target1 = s
target2 = s

count = 0
ans = 0
for i in range(n):
	if i % 2 == 0 and target1[i] == "1":
		count += 1
	elif i % 2 and target1[i] == "0":
		count += 1

	if i % 2 == 0 and target2[i] == "0":
		ans += 1
	elif i % 2 and target2[i] == "1":
		ans += 1
print(min(count, ans))



125
解説AC
[考察]
まず最大公約数を出すにはユークリッドの互除法を使う
a[i]を変えるときは、換えるのではなく消すと考える
前からのgcdと後ろからのgcdを持っておく
gcdの計算法則を利用した問題

ACだけどこれは互除法を使えていませんのでだめです
import fractions
def gcd(x,y):
    if x < y:
        x, y = y, x
    if y == 0:
        return x

    return fractions.gcd(x%y, y)

n = int(input())
a = list(map(int, input().split()))
L = [0]
R = [0]
now1 = 0
now2 = 0
ans = 0

for i in range(n-1):
    now1 = gcd(now1,a[i])
    L.append(now1)

    now2 = gcd(now2,a[-i-1])
    R.append(now2)

for i in range(n):
    ans = max(ans, gcd(L[i], R[n-i-1]))

print(ans)

CF
とても分かりやすい range(n)[::-1]にすると逆からいけるのは知見

N = int(input())
A = list(map(int, input().split()))

def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a%b)

L = [0] * (N+1)
R = [0] * (N+1)
for i in range(1, N+1):
    L[i] = gcd(L[i-1], A[i-1])
for i in range(N)[::-1]:
    R[i] = gcd(R[i+1], A[i])

M = [gcd(L[i], R[i+1]) for i in range(N)]
# max()はO(n)なので最後に一回だけがいいということでしょう

print(max(M))

126
[考察]
サンプルが詳しかったからそのまま実装するだけ
一回目でk以上の得点が出る場合に注意
n, k = map(int, input().split())
ans = 0

for i in range(1,n+1):
    point = 0
    if i >= k:
        ans += 1/n
    else:
        point = i
        count = 0
        while point < k:

            point *= 2
            count += 1
        ans += (1/n)*((1/2)**count)

print(ans)

127
[考察]
コードの通り
計算量を考える
リスト、セット関係なく計算量の勉強

n, m = map(int, input().split())
low = 0
high = n+1

for i in range(m):
    l, r = map(int, input().split())
    low = max(low, l)
    high = min(high, r)

print(high-low+1 if low <= high else 0)

128
[考察]
つながっているところの1の数とほかのスイッチと重なっているところは同じ
という考え方から、重なっているところは同じというのはbitに組み込まれているから
考えなくていいということになり、つながっているところの1の数%2が全てpと同じなら
条件を満たしているということで+1する
すべて同じならということは、一つでも違ったらだめなので、違ったときにflag == False
にしておく、違ったときのループからの抜け方が分からなかったので
n,m = map(int, input().split())
ks = [list(map(int, input().split())) for i in range(m)]
p = list(map(int, input().split()))
ans = 0
for switch in range(1 << n):
    target = str(bin(switch))[2:]
    if len(target) < n:
        target = "0"*(n-len(target)) + target
    flag = True
    for i in range(m):
        count = 0
        for j in ks[i][1:]:
            if target[j-1] == "1":
                count += 1

        if count % 2 != p[i]:
            flag = False
    if flag:
        ans += 1
print(ans)

129 ☆dp
[考察]
後ろから見ると、n-1番目からn番目に行くのは1通りの方法しかないとわかっている。
n-2番目からは、n-1+n通りの方法がある。同様にして、f(x) = f(x-1)+f(x-2)という関係式
が成り立っていることが分かる。このように、すでに決まっている情報からその情報に
依存している情報を求めたい、というときには、DP(動的計画法)が最適である。
この問題では、dp[n]=1,dp[a[i]]=0として、組み立てる。

[参考]
メモ化再帰という方法でもできるらしいが、今のところは解法が分からない
pythonでメモ化再帰を使うには、再帰関数の前に@lru_cache()と置く
from functools import lru_cache
@lru_cache()
def fibo(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibo(n-1) + fibo(n-2)
print(fibo(2)*fibo(2))

TLE
n,m = map(int, input().split())
a = [int(input()) for i in range(m)]
mod = (10**9)+7
dp = [1] + [0]*(n)
for i in range(1,n+1):
    if i in a:
        dp[i] == 0
    else:
        dp[i] = dp[i-1] + dp[i-2]
print(dp[-1]%mod)

AC
[注意]
python => pypy
input => sys.stdin.readline
!!!
ループが大きいときにlistに対してinは絶対にダメ
inはリストに対してとても遅く、set,dictに対してはリストの1/1000くらい
listは全探索、set,dictはハッシュテーブル(値とインデックスが入っているやつ)
大きいデータに対してやってはいけない書き方と対処法
a in b => setにするかbisectを使うか
a.remove() => collections.deque()というデータ構造でa.pop()
a.pop()

import sys
input = sys.stdin.readline
n,m = map(int, input().split())
A = set()
for i in range(m):
    a = int(input())
    A.add(a)
mod = (10**9)+7
dp = [1] + [0]*(n)
for i in range(1,n+1):
    if i in A:
    #リストに対するinは激遅なのでset
        dp[i] == 0
    else:
        dp[i] = dp[i-1] + dp[i-2]
        dp[i] %= mod
print(dp[-1])

130
[考察]
必ず面積を半分にする線分を引くことができる
その点が四角形の重心に位置するならば、複数の線が引ける、
そうでないならば一本しか引けない
・任意の点と中心をつなぐと、面積を半分に分割できる
∵合同な図形だから
w,h,x,y = map(int,input().split())
ans = (w*h)/2
if x == w/2 and y == h/2:
    print(ans,1)
else:
    print(ans,0)

131
[注意]
x以下でnで割り切れるものの個数はx//n (python)
lcm(x,y) = x*y // gcd(x,y)

WA 未解決
import math
import fractions
a,b,c,d = map(int, input().split())
e = a-1
less_than_e = e - math.floor(e/c) - math.floor(e/d) + math.floor(e/(c*d//fractions.gcd(c,d)))
less_than_b = b - math.floor(b/c) - math.floor(b/d) + math.floor(b/(c*d//fractions.gcd(c,d)))
print(less_than_b - less_than_e)

[考察]
[a,b]の範囲=>[0,a-1]と[0,b]で考えて引く

①
import fractions
a,b,c,d = map(int, input().split())
e = a-1
gcd = fractions.gcd(c,d)
less_than_e = e - (e//c) - (e//d) + (e//(c*d//gcd))
less_than_b = b - (b//c) - (b//d) + (b//(c*d//gcd))
print(less_than_b - less_than_e)

②
import fractions
a,b,c,d = map(int, input().split())
e = a-1
gcd = fractions.gcd(c,d)
def lcm(x, y):
    return (x * y) // gcd
lcm = lcm(c,d)
less_than_e = e - (e//c) - (e//d) + (e//lcm)
less_than_b = b - (b//c) - (b//d) + (b//lcm)
print(less_than_b - less_than_e)

132
[考察]
ソートして真ん中の二つの数の差
n = int(input())
d = sorted(list(map(int, input().split())))
print(d[int(n/2):][0]-d[:int(n/2)][-1])

133
WA
[考察]
範囲内に2019の倍数があればmod 2019== 0
なければ範囲の最小の二数の余りの掛け算
これをやって計算量を落とそうとしたけどそもそも考察が違ったし、
i%mod * j%mod != i*j%mod だった
l,r = map(int, input().split())
mod = 2019
for i in range(l,r+1):
    if i % 2019 == 0:
        print(0)
        exit()
print(l%mod * (l+1)%mod)
=>i%mod * j%mod != i*j%mod ?
間違えている理由①
2019 == 3*673だから、範囲の幅が2019より下でもmod2019が0になるときがある

WAだったのでやはりだめらしいもう少し勉強する必要がある
l,r = map(int,input().split())
r = min(r, l+673)
ans = 2018
for i in range(l,r):
    for j in range(l+1,r+1):
        if ans > (i%2019)*(j%2019):
            ans = (i%2019)*(j%2019)
print(ans)

python=>TLE,pypy=>AC
[考察]
mod2019の世界では、1=2019である.もしi<=jという制約ならば、mod2019の世界で同じ数を
選ぶことができるので、範囲をmin(r,l+2019)にすることで、二回も同じ数を探索しない
ようにすることができ、計算が間に合う.
i<jなので、mod2019の世界で同じ数を二回選んだ時のことを考慮して、min(r,l+4038)とすることで
同じ数を考慮した場合の全探索が可能になる.
ちなみに、2019の約数は{1,2019,3,673}なので、min(r,l+673)にすれば、この範囲で3の倍数が
あったときにi*j%2019==0になって少し探索時間が短くなる
l,r = map(int,input().split())
r = min(r, l+4038)
ans = 2018
for i in range(l,r):
    for j in range(l+1,r+1):
        if ans > i*j%2019:
            ans = i*j%2019
print(ans)

134
n = int(input())
a = [int(input()) for x in range(n)]
b = sorted(a,reverse=True)
for i in range(n):
    if b[0] == a[i]:
        print(b[1])
    else:
        print(b[0])

for i in range(len(a)):
    if a[:i] == []:
        print(max(a[i+1:]))
    elif a[i+1:] == []:
        print(max(a[:i]))
    else:
        print(max(max(a[i+1:]),max(a[:i])))


=>計算が多すぎるfor文1回ならまだ耐えられる
=>むやみやたらに提出しないで考える
=>その方法に固執しない

135
n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
sum_a = sum(a)

for i in range(n):
    if a[-1-i] - b[-1-i] >= 0:
        left = a[-1-i] - b[-1-i]
        a[-1-i] = left
        b[-1-i] = 0
    else:
        b[-1-i] = abs(a[-1-i] - b[-1-i])
        a[-1-i] = 0

    if a[-2-i] - b[-1-i] >= 0:
        a[-2-i] = a[-2-i] - b[-1-i]
    else:
        a[-2-i] = 0

print(sum_a - sum(a))

得点が同じなら時間最後に解いた問題の時間で決められる
最初のほうはcまでを早解きしたほうがレートは上がる
何度間違えてもその問題を解かなければ反映されない
けど解いたほうが、同じ正解数の人の中では下のほうになるけど順位はあがる
あくまでも得点が第一にあってどの問題を解いたか、どのくらいの時間で解いたかによって
その中で変動する
3完しても時間がごみ過ぎるとパフォーマンスは同じくらい
解く問題×速さ

136
<方針>
右の値を-1しても左のほうが小さかったら右の値を-1する
そうでなくて右の値そのままより左の値が小さかったらなにもしない
（-1したら左のほうが大きくなる可能性があるから）
何もしなくても左のほうが大きくなるなら（それ以外は）print("No")

n = int(input())
h = list(map(int, input().split()))
h[0] -= 1
for i in range(0, n - 1):
    diff = h[i + 1] - h[i]
    if diff >= 1:
        h[i + 1] -= 1
    elif diff < 0:
        print("No")
        exit()
print("Yes")
print(h)

WA
よく見たら全然違う
n = int(input())
h = list(map(int, input().split()))
for i in range(n-1):
    if h[i] > h[i+1]:
    # h[i] >= h[i+1]-1:
        h[i] = h[i]-1
        #h[i+1] -= 1
    #elif  h[i] >= h[i+1]:
        #continue
    #else:
        #print("No");exit()

if all(h[i] <= h[i+1] for i in range(n-1)):
        print("Yes")
        exit()
print("No")

137
ソートして今まで見たことあればvalueの回数 なければ辞書に追加
pypyよりpythonのほうが早かった
n = int(input())
d = {}
count = 0
for i in range(n):
    sli = sorted(list(input()))
    s = "".join(sli)
    if s in d:
        count += d[s]
        d[s] += 1
    else:
        d[s] = 1
print(count)

138
[考察]
配列が二つになるまで常に最も小さい二つの中間をとる
n = int(input())
v = sorted(list(map(int, input().split())))

while len(v) != 2:
    v = [(v[0]+v[1])/2] + v[2:]
    n = len(v)

print(sum(v)/2)

139
[考察]
やるだけ
n = int(input())
h = list(map(int, input().split()))
ans = 0
count = 0
for i in range(n-1):
    if h[i] >= h[i+1]:
        count += 1
    else:
        ans = max(ans, count)
        count = 0
ans = max(ans, count)
print(ans)

140
n = int(input())
b = list(map(int, input().split()))
ans = b[0] + b[-1]

for i in range(n-2):
    ans += min(b[i], b[i+1])
print(ans)

141
n,k,q = map(int, input().split())
li = [0]*n
for i in range(q):
		a = int(input())
		li[a-1] += 1
for i in li:
	if k - q + i <= 0:
		print("No")
	else:
		print("Yes")

142
[考察]
久しぶりかつ簡単すぎで焦った
タプルでインデックスとる

n = int(input())
a = list(map(int, input().split()))
li = []
for i in range(n):
    li.append((a[i], i+1))
li.sort()
for i in range(n):
    print(li[i][1])
