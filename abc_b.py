1
00は整数型だと0で出力される
m = int(input())/1000
if m < 0.1:
    print("00")
elif 0.1 <= m <= 5:
    print("{:02}".format(int(10*m)))
elif 6 <= m <= 30:
    print(int(m)+50)
elif 35 <= m <= 70:
    print(int((m-30)/5)+80)
elif 70 < m:
    print(89)

2
print(input().replace("a","").replace("i","").replace("u","").replace("e","").replace("o",""))

3
s = input()
t = input()
li = list("atcoder")
li.append("@")
for i in range(len(s)):
    if s[i]!=t[i] and s[i]!="@" and t[i]!="@":
        print("You will lose")
        exit()
    elif s[i]!=t[i] and (s[i] not in li or t[i] not in li):
        print("You will lose")
        exit()
print("You can win")

4 numpyで180度回転空白区切り出力
s = [list(input()) for i in range(4)]
import numpy as np
s=np.array(s)
s=s[::-1,::-1]
for i in s:
    print(" ".join(i))

cf for文で180°回転
s = [list(input().split()) for i in range(4)]
s = s[::-1]
for i in range(4):
    print(" ".join(s[i][::-1]))

5
n = int(input())
t = sorted([int(input()) for i in range(n)])
print(t[0])

6
やっぱり再帰では遅い、というか足し算中に余りを出す
n = int(input())
import sys
sys.setrecursionlimit(10**6)
mod = 10007
def tribo(n):
    if n == 1 or n == 2:
        return 0
    elif n == 3:
        return 1

    return tribo(n-1)%mod + tribo(n-2)%mod +tribo(n-3)%mod
print(tribo(n))

modの性質を使わないと計算オーバー
a,b,c=0,0,1
for i in range(int(input())-1):
    a,b,c = b,c,(a+b+c)%10007
print(a)

7
a = input()
print(-1 if a == "a" else "a")

8
n = int(input())
s = [input() for i in range(n)]
highest = 0
name = ""
for i in range(n):
    if s.count(s[i]) > highest:
        highest = s.count(s[i])
        name = s[i]
print(name)

9
n = int(input())
a = sorted(list(set([int(input()) for i in range(n)])))
print(a[-2])

10 等差数列
n = int(input())
a = list(map(int, input().split()))
ans = 0
for i in range(n):
    while a[i]%2==0 or (a[i]-2)%3==0:
        a[i]-= 1
        ans += 1
print(ans)

11
s = input()
print(s[0].upper() + s[1:].lower())

12
n = int(input())
h = n//3600
n -= 3600*h
m = n//60
n -= 60*m
ans = ""
for i in h,m,n:
    i = str(i)
    if len(i)==1:
        i = "0"+i
    ans += i+":"
print(ans[:-1])

cf format関数
N = int(input())
print("{:02}:{:02}:{:02}".format(N//3600,N%3600//60,N%60))

13
a = int(input())
b = int(input())
print(min(abs(a-b),10+min(a,b)-max(a,b)))

つまり
x = abs(int(input()) - int(input()))
print(min(x, 10 - x))

14 bit全探索入門の入門
n,x = map(int, input().split())
a = list(map(int, input().split()))
x = list(str(bin(x)))[2:][::-1]
ans = 0
for i in range(len(x)):
    if x[i] == "1":
        ans += a[i]
print(ans)

15
import math
n = int(input())
a = list(map(int, input().split()))
n -= a.count(0)
print(math.ceil(sum(a)/n))

16
a,b,c = map(int, input().split())
if a+b == c and a-b ==c :
    print("?")
elif a+b ==c:
    print("+")
elif a-b == c:
    print("-")
else:
    print("!")

17
すべて取り換えるなら省略可能
s = input()
s = s.replace("ch","").replace("o","").replace("u","").replace("k","")
if s == "":
    print("YES")
else:
    print("NO")

s = input()
s = s.replace("ch","").replace("o","").replace("u","").replace("k","")
if s == "":
    print("YES")
else:
    print("NO")

18 文字列の範囲指定からの反転に注意
s = input()
n = int(input())
for i in range(n):
    l,r = map(int, input().split())
    s = s[:l-1] + s[l-1:r][::-1] + s[r:]
print(s)

19 文字列の圧縮
果たしてこれを工夫というのか
s = input()+"0"
li = []
count = 1
for i in range(len(s)-1):

    if s[i]==s[i+1]:
        count +=1
    else:
        li.append(s[i])
        li.append(str(count))
        count = 1
s = "".join(li)
print(s)

想定解
s = input()
n = s[0]
count = 1
new_s = ''
for i in s[1:]:
    if i == n:
        count += 1
    else:
        new_s += n + str(count)
        n = i
        count = 1
new_s += s[-1] + str(count)
print(new_s)

20感謝
a,b = input().split()
print(int(a+b)*2)

21 ふしぎな問題
n = int(input())
a,b = map(int, input().split())
k = int(input())
p = list(map(int, input().split()))
p.append(a)
p.append(b)
for i in range(k+2):
    if p.count(p[i]) > 1:
        print("NO")
        exit()
print("YES")

22☆
根本的に問題を理解していないかった
その番目の前までにその種類の花があれば受粉する、なければここが最初と記録する
最初に考察もせず二重for文を書いたからそれに縛られた
n = int(input())
a = [int(input()) for i in range(n)]
a = set(a)
print(n-len(a))

23
a->b->cの順&奇数長&真ん中b
(n-1)/2番目の手順の直後に出る
n = int(input())
s = input()
if len(s)%2==0 or s[n//2]!="b":
    print(-1)
    exit()
for i in range(n-1):
    if s[i]=="b" and (s[i+1]!="c"):
        print(-1)
        exit()
    elif s[i]=="c" and (s[i+1]!="a"):
        print(-1)
        exit()
    elif s[i]=="a" and (s[i+1]!="b"):
        print(-1)
        exit()
条件超えたらこれだけ
print(int((n-1)/2))
この下必要ない
num = int((s.count("abc")+1)/2)
ans = 1 + 3*(num-1)
s = s.replace("abc","",s.count("abc"))
if len(s)==0:
    print(ans)
elif len(s)==2:
    print(ans+1)
else:
    print(ans+2)

24
n,t = map(int, input().split())
a = [int(input()) for i in range(n)]
count = t
for i in range(n-1):
    if a[i]+t > a[i+1]:
        count += a[i+1]-a[i]
    else:
        count += t
print(count)

25
n, a, b  = map(int, input().split())
add = 0
for i in range(n):
    direct, dist = input().split()
    dist = int(dist)
    if direct == "West":
        if dist < a:
            add -= a
        #いらない
        elif a <= dist <= b:
            add -= dist
        else:
            add -= b

    if direct == "East":
        if dist < a:
            add += a
        elif a <= dist <= b:
            add += dist
        else:
            add += b

if add < 0:
    print("West",abs(add))
elif add > 0:
    print("East",add)
else:
    print(0)

cf
n, a, b = [int(el) for el in input().split(' ')]
c = 0
for i in range(n):
    s, d = input().split(' ')
    d = int(d)

    if d < a:
        d = a
    elif d > b:
        d = b

    if s == 'East':
        d = -d
    c += d

if c < 0:
    print('East {}'.format(abs(c)))
elif c == 0:
    print(0)
else:
    print('West {}'.format(abs(c)))

26 math.pi
n = int(input())
r = sorted([int(input()) for i in range(n)],reverse = True)
red = r[::2]
white = r[1::2]
rad = 0
for i in  range(len(red)):
    rad += (red[i])**2
for j in range(len(white)):
    rad -= (white[j])**2
import math
print(rad*math.pi)

27☆☆
左から数えてそこまでの総和で平均にできないなら橋を架ける
思いつかない
n = int(input())
a = list(map(int, input().split()))
if sum(a)%n:
    print(-1)
    exit()
mean = int(sum(a)/n)
count = 0
for i in range(n-1):
    if sum(a[:i+1]) != mean*(i+1):
        count += 1
print(count)

28 空白区切りで出力
li = []
s = input()
string = ["A","B","C","D","E","F"]
for i in range(6):
    li.append(s.count(string[i]))
print(' '.join(map(str,li)))

cf
s = input()
print(*[s.count(c) for c in "ABCDEF"], sep=" ")


29
count = 0
for i in range(12):
    s = input()
    if "r" in s:
        count += 1
print(count)

30 考察が長い
n, m = map(int, input().split())
if n >= 12:
    n -= 12
r1 = 30*n + 0.5*m
r2 = 6*m
ans = max(r1,r2)-min(r1,r2)
print(360 - ans if ans > 180 else ans)

31
l,h = map(int, input().split())
n = int(input())
for i in range(n):
    a = int(input())
    if l <= a <= h:
        print(0)
    elif a < l:
        print(l-a)
    else:
        print(-1)

32
s = input()
k = int(input())
if len(s) < k:
    print(0)
    exit()
li = []
for i in range(len(s)-k+1):
    li.append(s[i:i+k])
print(len(set(li)))

33 文字列と数字を同時に扱いたいときは辞書型を考える
n = int(input())
ans = 0
li = {}
next = ""
for i in range(n):
    name, people = input().split()
    li.setdefault(name,int(people))
    ans = sum(li.values())
    if int(ans/2) < li[name]:
        next = name

print(next if int(ans/2) < li[next] else "atcoder")

34
n = int(input())
print(n-1 if n%2==0 else n+1)

35 TLEだけど形にはなった 跡は計算量を落とすテクニック

s = list(input())
t = int(input())
x = s.count("R") - s.count("L")
y = s.count("U") - s.count("D")
n = s.count("?")
ans = []
for r in range(n+1):
    for l in range(n+1-r):
        for u in range(n+1-r-l):
            d = n-r-l-u
            ans.append(abs(x+r-l)+abs(y+u-d))
if t == 1:
    print(max(ans))
else:
    print(min(ans))

最大値と最小値だけを考える、最大の場合はコードの通り
最小の時は(0,0)の時に動くことしかできないので、何往復もして何とか(0,0)に近づくように
努力する
s = input()
t = int(input())
x = s.count("R") - s.count("L")
y = s.count("U") - s.count("D")
n = s.count("?")

if x >= 0 and t == 1:
    print(x + n + abs(y))
elif x < 0 and t == 1:
    print(abs(x - n) + abs(y))


if t == 2 and n <= abs(x) + abs(y):
    print(abs(x) + abs(y) - n)
elif t == 2 and abs(x) + abs(y) < n:
    n -= abs(x) + abs(y)
    if n%2:
        print(n%2)
    else:
        print(0)

36 時計回りはnumpy numpyは遅すぎる
import numpy as np
n = int(input())
s = [list(input()) for i in range(n)]
s = np.array(s)
s = s[::-1,:].T
for i in s:
    print("".join(i))

②
import numpy as np
n = int(input())
s = [list(input()) for i in range(n)]
s = np.array(s)
s = np.rot90(s,-1)
for i in s:
    print("".join(i))

③zipで時計回りに90度回転 圧倒的な速さ
n = int(input())
s = [list(input()) for i in range(n)]
sT = zip(*s[::-1])
for i in sT:
    print("".join(i))

37 前に参考解答で見たような書き方
n,q = map(int, input().split())
li = ["0"]*n
for i in range(q):
    l,r,t = map(int, input().split())
    li = li[:l-1] + [str(t)]*(r-l+1) + li[r:]

for i in range(n):
    print(li[i])

38
h1,w1 = map(int, input().split())
h2,w2 = map(int, input().split())
print("YES" if h1 == h2 or h1 == w2 or w1 == h2 or w1 == w2 else "NO")

39
n = int(input())
print(int(n**(1/4)))

40 平方数なら0,それ以外はn以下の最大の平方数との差と、平方数以外かつ約数が2以上の
正の数の最も約数同士の差が小さいものの差の全探索で最小値

pypyだと実行時間1/4,メモリ7倍

import math
n = int(input())
if math.sqrt(n).is_integer():
    print(0)
    exit()

def make_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n//i)

    return divisors

li = []
i = n
while math.sqrt(i).is_integer() == False:
    i -= 1
else:
    li.append(n-i)
for j in range(2,n+1):
    nli = make_divisors(j)
    if len(nli)%2:
        continue
    else:
        li.append(n - j + nli[-1] - nli[-2])

print(min(li))

想定解
縦×横を全探索して面積がn以上になったら終わり 最小値
ans=[]
n=int(input())
for i in range(1,n+1):
	for j in range(1,n+1):
		a=i*j
		if a>n:
			break
		else:
			ans.append(n-a+abs(i-j))
print(min(ans))


41
足し算、引き算、掛け算でmodが取れるときはとる
a,b,c = map(int, input().split())
mod = (10**9)+7
print((a*b%mod)*c%mod)

42
n,l = map(int, input().split())
li = sorted([input() for x in range(n)])
s = ""
for i in range(n):
    s += li[i]
print(s)

43
s = list(input())
str = ""
for i in range(len(s)):
    if s[i] == "0":
        str += "0"
    elif s[i] == "1":
        str += "1"
    else:
        if str == "":
            continue
        else:
            str = str[:-1]
print(str)
#if if if => if elif else
#最後からなら-1
#Bスタートの時WA
cf.
s = input()
a = ''
for x in s:
    if x == '0':
        a += '0'
    elif x == '1':
        a += '1'
    else:
        if a != '':
            a = a[:-1]
print(a)

44
w = list(input())
for i in range(len(w)):
    if w.count(w[i]) % 2:
        print("No")
        exit()
print("Yes")

45
card = [list(input()) for i in range(3)]
main = card[0]
target = "A"
while len(main) != 0:
    if main[0]=="a":
        main.remove(main[0])
        main = card[0]
        target = "A"
    elif main[0]=="b":
        main.remove(main[0])
        main = card[1]
        target = "B"
    else:
        main.remove(main[0])
        main = card[2]
        target = "C"

print(target)

cf
S={c:list(input()) for c in "abc"}
s="a"
while S[s]:
    s = S[s].pop(0)
print(s.upper())

46
n,k = map(int, input().split())
print(int(k*((k-1)**(n-1))))

47 最初に書いた条件であっていたのに直してしまった
書く前に気を付けなければいけない場面を意識
w,h,n = map(int, input().split())
li = [0, w, 0, h]
ans = 0
for i in range(n):
    x,y,a = map(int, input().split())
    if a == 1 and li[0] < x:
        li[0] = x
    elif a == 2 and li[1] > x:
        li[1] = x
    elif a == 3 and li[2] < y:
        li[2] = y
    elif a == 4 and li[3] > y:
        li[3] = y
W = li[1]-li[0]
H = li[3]-li[2]
print(0 if W < 0 or H < 0 else W*H)

48
a, b, x = map(int, input().split())
print(b // x - (a - 1) // x)
数列的な考え方
全体の個数から部分をとる

49
h,w = map(int,input().split())
#row == 行 column == 列
matrix = []
while True:
    try:
        matrix.append(list(map(str,input().split())))
    except:
        break;
for i in range(len(matrix)):
    print(matrix[i][0])
    print(matrix[i][0])

50
n = int(input())
t0 = list(map(int, input().split()))
m = int(input())
px = []
while True:
    try:
        px.append(list(map(int, input().split())))
    except:
        break;
import copy

for i in range(m):
    #shallow copy , deep copy の話
    t = []
    t = copy.copy(t0)
    t[px[i][0]-1] = px[i][1]
    print(sum(t))

B問題しっかりやると勉強になる
やはり自分でACしようとするといろいろと調べる

cf

n = int(input())
t = list(map(int, input().split()))
total = sum(t)
m = int(input())
for i in range(m):
    p, x = map(int, input().split())
    print(total - t[p - 1] + x)

 複数行の時にfor文の中で受け取るという考え方

51
k,s = map(int,input().split())
count = 0
for x in range(k+1):
    for y in range(k+1):
        z = s - x - y
        if 0 <= z <= k:
            count += 1
print(count)

1秒間で処理できるfor文ループの回数は10の8乗回

52
n = int(input())
s = list(input())
x = [0]
now = 0
for i in range(n):
    if s[i] == "I":
        now += 1
        x.append(now)
    elif s[i] == "D":
        now -= 1
        x.append(now)
print(max(x))

53
s = input()
a = s.index("A")
s = s[::-1]
z = s.index("Z")
print(len(s)-z-a)

54☆☆
n,m = map(int, input().split())
a = [input() for i in range(n)]
b = [input() for i in range(m)]

for i in range(m):
    if b[i] in a[i][:len(b[i])]:
        print("Yes")
        exit()
print("No")
保留

これだと平行移動左上だけ
n, m = map(int, input().split())
a = [list(input()) for x in range(n)]
for i in range(m):
    target = list(input())
    if any(target[j] == a[i][j] for j in range(m)):
        print("Yes")
        exit()
print("No")
保留

AC 9/1
n,m = map(int, input().split())
a = [list(input()) for i in range(n)]
b = [list(input()) for i in range(m)]

for i in range(n):#左上の列を固定
    if n < i + m:
        break
    for j in range(n):#左上の行を固定
        if n < j + m:
            break
        if all(a[j+k][i:i+m] == b[k] for k in range(m)):
            #スタート行からm進んだところまでですべて同じなら〇
            print("Yes")
            exit()
print("No")

55
import math
n = int(input())
print(int(math.factorial(n)%(10**9+7)))
実装に悩んでいる暇があるなら積極的に調べる
考察に時間を使うのはいいけど実装に時間を使うのはあほ
有名な計算であれば最適化されたモジュールがあるはず

cf
n = int(input())
def power(n):
    if n < 1:
        return 1

    return n * power(n-1)
print(int(power(n)%(10**9+7)))
速度やばい

56
w,a,b = map(int, input().split())
if a <= b <= a+w or a <= b+w <= a+w:
    print(0)
elif a+w < b:
    print(b-(a+w))
elif b+w < a:
    print(a-(b+w))

57
n,m = map(int, input().split())
ab = [list(map(int, input().split())) for i in range(n)]
cd = [list(map(int, input().split())) for j in range(m)]
for i in range(n):
    dist = []
    x = ab[i][0]
    y = ab[i][1]
    for j in range(m):
        dist.append(abs(x - cd[j][0]) + abs(y - cd[j][1]))

    print(dist.index(min(dist))+1)

エラーがあったら該当箇所をよく見直すこと
何が問われているのかを整理すること

58
o = input()
e = input()
li = "".join([i+j for (i,j) in zip(o,e)])
if len(o) == len(e):
    print(li)
elif len(o) > len(e):
    print(li + o[-1])
else:
    print(li + e[-1])

始めてタプルを使った
"".join()の考え方
二つを同時に扱うときにタプルでzip

cf
o = input()
e = input()

s = ""
if len(o) == len(e):
    for i in range(len(o)):
        s += o[i]+e[i]
else:
    for i in range(len(o)-1):
        s += o[i]+e[i]
    s += o[-1]
print(s)
標準的な回答
制約 len(o) - len(e) == 0 or 1 より len(o)が1多いときとで場合分け
制約をしっかり見ましょう

59
a = int(input())
b = int(input())
if a > b:
    print("GREATER")
elif a < b:
    print("LESS")
else:
    print("EQUAL")

60☆
数学的な問題 お手上げ
n番目を考えて式を変形する
a,b,c = map(int, input().split())
for i in range(1,b+1):
    if (i*a)%b == c:
        print("YES")
        exit()
print("NO")

61辞書すごく便利
AC
n, m = map(int, input().split())
ans = dict.fromkeys([str(i) for i in range(1,n+1)], 0)
li = [list(map(int, input().split())) for j in range(m)]
for i in range(m):
    ans[str(li[i][0])] += 1
    ans[str(li[i][1])] += 1
for k in range(1,n+1):
    print(ans[str(k)])

別の解法
長さNので全要素0のリストを作るm回でループ回してa,b=map(int, input().split())
li[a-1]とli[b-1]を+1する
つまり
n,m = map(int, input().split())
li = [0]*n
for i in range(m):
    a,b = map(int, input().split())
    li[a-1]+=1
    li[b-1]+=1
for j in range(n):
    print(li[j])

WA
n,m = map(int, input().split())
li = [list(map(int, input().split())) for i in range(m)]
li = [flatten for inner in li for flatten in inner ]
a = li[::2]
b = li[1::2]
for i in range(m):

        if a.count(i+1) + b.count(i+1) == 0:
            exit()
        else:
            print(a.count(i+1) + b.count(i+1))

62 周辺埋め 10min
h, w = map(int, input().split())
s = ["#"*(w+2)]+["#"+input()+"#" for i in range(h)]+["#"*(w+2)]
for i in range(len(s)):
    print(s[i])

63 全探索 3min
s = list(input())
for i in range(len(s)):
    if s.count(s[i]) >= 2:
        print("no")
        exit()
print("yes")

64 基本 7min
n = int(input())
li = sorted(list(map(int, input().split())),reverse = True)
ans = []
for i in range(n-1):
    ans.append(li[i] - li[i+1])
print(sum(ans))

65☆☆
AC 全部一回しか通っちゃダメ=>最大で全部通ってn-1回=>n-1回まででだめならダメ
n = int(input())
a = [int(input()) for i in range(n)]
count = 1
now = a[0]
while now != 2 and count < n:
    count += 1
    now = a[now-1]
print(count if count < n else -1)

TLE
n = int(input())
a = [int(input()) for i in range(n)]
on = 1
count = 0
on_li = [1]
while True:
    on = a[on-1]
    count += 1
    if on == 2:
        print(count)
        exit()

    if on in on_li:
        print(-1)
        exit()
    else:
        on_li.append(on)

66 午前二時は頭が動かん 文字列操作
s = list(input())
s = s[:len(s) - 1]
for i in range(len(s)):

    if s[:int(len(s)/2)] == s[int(len(s)/2):]:
        print(len(s))
        exit()
    else:
        s = s[:-1]

67 基本 5min
n, k = map(int, input().split())
l = sorted(list(map(int, input().split())))
print(sum(l[n-k:]))

68 基本 実装の仕方多分いろいろ
li = [1,2,4,8,16,32,64]
n = int(input())
if n in li:
    print(n)
    exit()

li.append(n)
li.sort()
num = li.index(n)
print(li[num-1])

69 文字列の連結 3min
s = input()
print(s[0] + str(len(s[1:len(s)-1])) + s[-1])

70
a,b,c,d = map(int, input().split())
alice = list(range(a,b))
bob = list(range(c,d))
ans = 0

if len(alice) <= len(bob):
    target = alice
    other = bob
else:
    target = bob
    other = alice

for i in range(len(target)):
    if target[i] in other:
        ans += 1
print(ans)

cf これが見えたら強い
A, B, C, D = map(int, input().split())
print(max(0, min(B, D) - max(A, C)))

71  文字列操作 アルファベット 6min
import string
li = string.ascii_lowercase
s = sorted(list(input()))
for i in range(26):
    if not li[i] in s:
        print(li[i])
        exit()
print("None")

72 1min
print(input()[::2])

73 ここを瞬殺したのは大きい 3min
n = int(input())
ans = 0
for i in range(n):
    l, r = map(int, input().split())
    ans += (r-l+1)
print(ans)

74 問題文が長いだけ
n = int(input())
k = int(input())
x = list(map(int, input().split()))
li = []
for i in range(n):
    li.append(min(2*x[i], 2*(k-x[i])))
print(sum(li))

75
非常に難しかったけど勉強になった
for文の中で条件を付けることで四隅をクリアする
h,w = map(int, input().split())
s = [list(input().replace(".","0")) for i in range(h)]
x = [1,1,0,-1,-1,-1,0,1]
y = [0,1,1,1,0,-1,-1,-1]

for i in range(h):
    for j in range(w):
        if s[i][j] == "0":
            for k in range(8):
                if 0 <= j+y[k] < w and 0 <= i+x[k] < h and s[i+x[k]][j+y[k]] == "#":
                    s[i][j] = str(int(s[i][j])+1)

for i in range(h):
    print("".join(s[i]))

76 4min
n = int(input())
k = int(input())
now = 1
for i in range(n):
    now = min(now*2, now+k)
print(now)

77 ルートして整数にして二乗 10min
n = int(input())
print(int(n**(1/2))**2)

78 while 6min
x,y,z = map(int,input().split())
x -= 2 * z
count = 0
while x >= y:
    x -= (y+z)
    count += 1
else:
    print(count)

79 添え字が課題 インクリメントも 20min
n = int(input())
li = [2,1]
for i in range(n+1):
    if i == 0:
        continue
    elif i == 1:
        continue
    else:
        li.append(li[i-2] + li[i-1])
print(li[n])

cf(TLE)
def ryuka(n):
    if n == 0:
        return 2
    elif n == 1:
        return 1

    return ryuka(n-1) + ryuka(n-2)
print(ryuka(n))

80 桁区切り 6min
n = int(input())
li = list(str(n))
a = sum([int(li[i]) for i in range(len(li))])
print("Yes" if n%a == 0 else "No")

81 この制約ならTLEのはずなのに通る
n = int(input())
a = list(map(int, input().split()))
count = 0
while all(i%2 == 0 for i in a):
    a = [j/2 for j in a]
    count += 1
print(count)

82 3min 文字列は比較できる(辞書順か)
s = sorted(list(input()))
t = sorted(list(input()), reverse = True)
print("Yes" if s < t else "No")

83 添え字は数をこなして慣れるしかない 10min
li = []
n,a,b =map(int, input().split())
for i in range(n+1):
    c = list(str(i))
    d = sum([int(c[x]) for x in range(len(c))])

    if a <= d <= b:
        li.append(i)
print(sum(li))

84 8min 工夫
a,b = map(int, input().split())
s = input()
if s[a] == "-" and s.count("-") == 1:
    print("Yes")
else:
    print("No")

85
n = int(input())
li = set([int(input()) for i in range(n)])
print(len(li))

86 平方数の判定
a,b = map(str, input().split())
num = int(a + b)
if int(num**(1/2))**2 == num:
    print("Yes")
else:
    print("No")

87 三重のfor文でないとできない 3min
a = int(input())
b = int(input())
c = int(input())
x = int(input())
ans = 0
for i in range(a+1):
    for j in range(b+1):
        for k in range(c+1):
            if 500*i + 100*j + 50*k == x:
                ans += 1
print(ans)

88  ソートとスライス 2min
n = int(input())
a = sorted(list(map(int, input().split())), reverse = True)
print(sum(a[::2]) - sum(a[1::2]))

89 set()必要ない
n = int(input())
s = set(list(map(str, input().split())))
print("Four" if "Y" in s else "Three")

90 逆順[::-1]
a,b = map(int, input().split())
ans = 0
for i in range(a,b+1):
    s = str(i)
    t = s[::-1]
    if s[0:2] == t[0:2]:
        ans += 1
print(ans)

91 負の数と比較したいときにmax(0,num)を使う
n = int(input())
s = sorted([input() for i in range(n)])
m = int(input())
t = [input() for j in range(m)]
li = []
for x in range(n):
    li.append(s.count(s[x]) - t.count(s[x]))
print(max(max(li),0))

92☆
苦手なのは添え字じゃなくてforとwhileの組み合わせかも whileでも
インクリメントの変数を入れるところが特に 大きな一歩
n = int(input())
d,x = map(int, input().split())
a = [int(input()) for i in range(n)]
for i in range(n):
    k = 0
    count = 0
    while k*a[i]+1 <= d:
        k += 1
        count += 1
    else:
        x += count
print(x)

93☆☆ この解法がダメな理由は強くなった時に
保留
a,b,k = map(int,input().split())
li = list(range(a,b+1))
list_ = li[:k] + li[-k:]
list_ = sorted(list(set(list_)))
for i in range(len(list_)):
    print(list_[i])

a,b,k = map(int, input().split())
li = list(range(a,b+1))
li_low = li[:k]
li_high = li[-k:]
li_ans = sorted(list(set(li_low + li_high)))
for i in range(len(li_ans)):
    print(li_ans[i])

想定解
a,b,k = map(int,input().split())
if (b-a+1) <= 2*k:
    for k in range(a,b+1):
        print(k)
else:
    for i in range(a,a+k):
        print(i)
    for j in range(b-k+1,b+1):
        print(j)

94 複雑そうに見えて問われているのは単純なこと
n,m,x = map(int, input().split())
a = list(map(int, input().split()))
b = 0
c = 0
for i in range(m):
    if 0 < a[i] < x:
        b += 1
    elif x < a[i] < n:
        c += 1
print(min(b,c))

95
n,x = map(int, input().split())
m = sorted([int(input()) for i in range(n)])
x -= sum(m)
add = x // m[0]
print(n + add)

96 リストの要素
li =sorted(list(map(int, input().split())))
k = int(input())
print(sum(li[:-1]) + li[-1]*(2**k))

97☆for と while が混ざるとわからなくなる
x = int(input())
if x == 1:
    print(1)
    exit()
li = list(range(2,x+1))
ans = [1]
n = 2
while li[0]**n < x:
    for i in range(x-1):
        if li[i]**n > x:
            ans.append(li[i-1]**n)
            break
    n += 1
print(max(ans))

98 流れが悪いと時間がかかる 集合はsetオブジェクト
n = int(input())
s = list(input())
li = []
for i in range(1,n-1):
    li.append(len(set(s[:i+1]) & set(s[i+1:])))
print(0 if len(li) == 0 else max(li))

cf 20190819
n = int(input())
s = input()
ans = 0
for i in range(1,n-1):
    count = 0
    a = list(set(s[:i]))
    b = list(set(s[i:]))
    for j in range(len(a)):
        if a[j] in b:
            count += 1
    if ans < count :
        ans = count

print(ans)

99☆ 保留 再帰関数
下のと同じことをやっているはずなのにRe
a,b = map(int, input().split())
diff = b - a - 1
def tower(n):
    if n < 1:
        return 0
    return n + tower(n - 1)
print(tower(diff) - a)

cf
a,b = map(int, input().split())
print(int(((b-a)*(b-a+1))/2) - b)

100☆
WA
d, n = map(int, input().split())
print(int(n*100**d))

n == 100 のとき d+1回割り切れることになるので、その次の数
d, n = map(int, input().split())
if n == 100:
    n += 1
print((100**d)*n)

101 2min
n = int(input())
li = list(str(n))
a = [int(li[i]) for i in range(len(li))]
print("No" if n % sum(a) else "Yes")

102 2重ループすると計算間に合わない
n = int(input())
a = sorted(list(map(int, input().split())))
print(a[-1] - a[0])

103 文字列の連結 "".join() rotate関数は自前
s = list(input())
t = input()
li = []
def rotate(l, n):
    return l[n:] + l[:n]
for i in range(len(s)):
    li.append("".join(rotate(s, i)))
print("Yes" if t in li else "No")

104問題文をしっかり読みましょう あと人の解答も見ましょう
import string
li = list(string.ascii_lowercase)
s = list(input())
if s[0] == "A" and s[2:-1].count("C") == 1:
    s.remove("C")
    s.remove("A")
else:
    print("WA")
    exit()
if all([s[i] in li for i in range(len(s))]):
    print("AC")
    exit()
else:
    print("WA")

cf replace(検索,置換,個数) str.islower(), str.isupper()
i = input()
if i[0]=="A"and i[2:-1].count("C")==1\
and i[1:].replace("C","c",1).islower()==True:
    print("AC")
else:print("WA")

105 問題文を本当にしっかり読んでください
n = int(input())
c = [i for i in range(0,n+1) if i%4 == 0]
d = [j for j in range(0,n+1) if j%7 == 0]
for x in range(len(c)):
    for y in range(len(d)):
        if c[x] + d[y] == n:
            print("Yes")
            exit()
print("No")

106 約数列挙からリストの長さ
n = list(range(1,int(input())+1,2))
li = []
def make_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n//i)

    # divisors.sort()
    return divisors
for i in range(len(n)):
    if len(make_divisors(n[i])) == 8:
        li.append(n[i])
print(len(li))

想定解 nの約数=range(1,n+1)で割り切れた個数
n = int(input())
ans = 0
for i in range(1,n+1,2):
    count = 0
    for j in range(1,i+1):
        if i%j == 0:
            count += 1
            if count == 8:
                ans += 1
print(ans)

107 メモは直したところ 考え方はあっている
h,w = map(int, input().split())
li = []
for i in range(h):
    a = list(input())
    '''
    if all([a[i] == "." for i in range(w)]):
        continue
    else:
        li.append(a)
    もう少し考えを柔軟に
    '''
    if "#" in a:
      li.append(a)

for j in range(w):
    #if all([li[k][j] == "." for k in range(len(li))]):
    #all関数はこの場面はリスト内包表記ではなくジェネレータ式
    #関数の唯一の引数として呼び出される場合は外側の丸かっこを省略できる
    #という構文上の決まりがあります。
    '''
    どんな関数に対して有用かと言うと、リスト内包表記の結果生成される
    listがなくても構わないような処理をする関数に対して威力を発揮します。
    逆に言うと、それ以外で使うのは反則です。
    all(),any()でよく使う
    '''
    if all(li[k][j] == "." for k in range(len(li))):
        for k in range(len(li)):
            #li[k][j] == ""
            li[k][j] = ""
for m in range(len(li)):
    print("".join(li[m]))

解法② 回転させる zip(*)で90度回転(反時計回り)
h, w = map(int, input().split())
a = [list(input()) for i in range(h)]
a = [x for x in a if "#" in x]
aT = list(zip(*a))
aT = [x for x in aT if "#" in x]
a = list(zip(*aT))
for i in a:
    print("".join(i))

108
x1,y1,x2,y2 = map(int, input().split())
print(x2-(y2-y1), y2+(x2-x1), x1-(y2-y1), y1+(x2-x1))

109
n = int(input())
w = [input() for i in range(n)]
for j in range(len(w)):
    if w.count(w[j]) > 1:
        print("No")
        exit()
for k in range(len(w)-1):
    if w[k][-1] != w[k+1][0]:
        print("No")
        exit()
print("Yes")

110 制約と誓約
n,m,x,y = map(int, input().split())
xli = list(map(int, input().split()))
yli = list(map(int, input().split()))
xmax = max(xli)
ymin = min(yli)
for z in range(x+1, y+1):
    if xmax < z <= ymin:
        print("No War")
        exit()
print("War")

111 pythonistaの解答
n = input()
while n.count(n[0]) != 3:
    n = int(n)
    n += 1
    n = str(n)
print(n)

cf 111
n = int(input())
i = int(n / 111)
if n % 111 != 0:
    i += 1
print(i * 111)

112 いい加減問題文をよく読んでください
n,T = map(int, input().split())
tli = []
cli = []
for i in range(n):
    c,t = map(int, input().split())
    if t <= T:
        tli.append(t)
        cli.append(c)
print("TLE" if tli == [] else min(cli))

113 WAの理由が分からないときはどうしようもない
WA
n = int(input())
t,a = map(int, input().split())
h = list(map(int, input().split()))
li = []
for i in range(n):
    diff = abs(int(t-(h[i]*0.006))-a)
    li.append(diff)
print(li)
print(li.index(min(li))+1)

AC
int()を外したらACになった int()したことで繰り上がったことがあったのだろう
n = int(input())
t,a = map(int, input().split())
h = list(map(int, input().split()))
li = []
for i in range(n):
    diff = abs((t-(h[i]*0.006))-a)
    li.append(diff)

print(li.index(min(li))+1)

114
s = input()
li = []
for i in range(len(s)-2):
    num = int(s[i:i+3])
    li.append(abs(num - 753))
print(min(li))

115
n = int(input())
p = sorted([int(input()) for i in range(n)])
print(int(sum(p[:-1])+(p[-1]/2)))

116 添え字は丁寧に扱う
while True: => 条件に一致したらループを抜ける
s = int(input())
li = [s]
i = 1
while  True:
    if li[i-1]%2==0:
        li.append(int(li[i-1]/2))
    else:
        li.append(3*li[i-1]+1)

    if li[i] in li[:i]:
        break
    else:
        i += 1
print(len(li))

117 真に短い => 等号つかない不等号
n = int(input())
l = sorted(map(int, input().split()))
print("Yes" if l[-1] < sum(l[:-1]) else "No")

118 積集合
n,m = map(int, input().split())
ans = list(input().split())
ans = set(ans[1:])
for i in range(n-1):
    b = list(input().split())
    b = set(b[1:])
    ans = ans & b
print(len(ans))

119
n = int(input())
ans = 0
for i in range(n):
    money, kind = map(str, input().split())
    if kind == "JPY":
        ans += float(money)
    else:
        ans += float(money)*380000.0
print(ans)

120 約数列挙で積集合(セットオブジェクト)
a,b,k = map(int, input().split())
def make_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n//i)

    # divisors.sort()
    return divisors
ali = set(make_divisors(a))
bli = set(make_divisors(b))
li = sorted(list(ali & bli))
print(li[-k])

121 if文の位置に注意
n,m,c = map(int, input().split())
b = list(map(int, input().split()))
count = 0
for i in range(n):
    a = list(map(int, input().split()))
    li = []
    for j in range(m):
        li.append(a[j]*b[j])
    if sum(li) + c > 0:
            count += 1
print(count)

122
最初 ans.append()をループ中に入れていたのでいちいちappend()された
最後続いて終わったときの処理に注意
s = input()
li = ["A", "T", "G", "C"]
count = 0
ans = []
for i in range(len(s)):
    if s[i] in li:
        count += 1
    else:
        ans.append(count)
        count = 0
ans.append(count)
print(max(ans))

123 正確に書けるようになってきた
num より大きくて一番近い10の倍数との差が最も大きいものを最後に注文
li = []
ans = 0
for i in range(5):
    num = int(input())
    ans += num
    if num % 10 == 0:
        continue
    else:
        p = num//10
        li.append((10*(p+1))-num)
li = sorted(li)
print(ans + sum(li[:-1]))

124 そこの前までのリストの最大以上ならカウント
n = int(input())
h = list(map(int, input().split()))
ans = 1
for i in range(1,n):
    li = h[:i]
    if max(li) <= h[i]:
        ans += 1
print(ans)

125 コスパがプラスだったら足す
n = int(input())
v = list(map(int, input().split()))
c = list(map(int, input().split()))
li = [v[i]-c[i] for i in range(n)]
ans = 0
for j in range(n):
    if li[j] >= 0:
        ans += li[j]
print(ans)

126☆
AC
s = input()
s1 = int(s[0:2])
s2 = int(s[2:])
if 1 <= s1 <= 12 and 1 <= s2 <= 12:
    print("AMBIGUOUS")
elif 1 <= s1 <= 12:
    print("MMYY")
elif 1 <= s2 <= 12:
    print("YYMM")
else:
    print("NA")

WA NAの理由以外は同じ仕様=>NAの理由はほかにもある
s = input()
s1 = s[0:2]
s2 = s[2:]
m = ["01","02","03","04","05","06","07","08","09","10","11","12"]
x = [str(i) for i in range(13,100)]
if (s1 in m and s2 in m) :
    print("AMBIGUOUS")
elif (s1 in x and s2 in x) or ("00" in s1 or "00" in s2 ):
    print("NA")
if s1 in m and s2 in x:
    print("MMYY")
elif s1 in x and s2 in m:
    print("YYMM")

127
r,d,x = map(int, input().split())
for i in range(10):
    x = r*x - d
    print(x)

128
当然WA
n = int(input())
li = [list(map(str, input().split())) for i in range(n)]
[li[i].append(str(i+1)) for i in range(n)]
li = sorted(li, key=lambda x:x[0])
rest = []
ans = []
for j in range(n-1):
    if li[j][0] == li[j+1][0]:
        rest.append(li[j])

    else:
        rest.append(li[j])
        ans.append(sorted(rest, key=lambda x:x[1], reverse = True))
        rest = []
rest.append(li[j+1])
ans.append(sorted(rest, key=lambda x:x[1], reverse = True))
ans = [flatten for inner in ans for flatten in inner]
for k in range(n):
    print(int(ans[k][2]))

AC 学びの多い一問
n = int(input())
a = [input().split() +[i+1] for i in range(n)]
#[i+1]にすることでリストで連結
'''
[['khabarovsk', '20', 1], ['moscow', '10', 2], ['kazan', '50', 3],
 ['kazan', '35', 4], ['moscow', '60', 5], ['khabarovsk', '40', 6]]
'''
a = sorted(a ,key=lambda x:(x[0], -int(x[1])))
#-int(x[1]) => マイナスつけて負の数の昇順==正の降順
#lambda 複数キーでのソート(先のキーが優先)

'''
もしくは
from operator import itemgetter
a = [ [1, 2, 3], [1, 1, 4], [4, 5, 2] ]
print(sorted(a, key=itemgetter(1)))
# [ [1, 1, 4], [1, 2, 3], [4, 5, 2] ]

[['kazan', '50', 3], ['kazan', '35', 4], ['khabarovsk', '40', 6],
['khabarovsk', '20', 1], ['moscow', '60', 5], ['moscow', '10', 2]]
'''
for i in range(len(a)):
    print(a[i][2])

129
n = int(input())
w = list(map(int, input().split()))
li = []
for i in range(1,n):
    li.append(abs(sum(w[:i])-sum(w[i:])))
print(min(li))

130
WA
これだと最後までx以下だった時に1少ない
n,x = map(int, input().split())
l = list(map(int, input().split()))
count = 0
now = 0
for i in range(n):
    if now <= x:
        now += l[i]
        count += 1
    else:
        break
print(count)

ex)
input
4 12
3 3 3 3
output
4 #本当は5

131
AC
n,l = map(int, input().split())
remove = l
li = [l]
for i in range(2,n+1):
    if abs((l+i-1)-0) < abs(remove):
        remove = l+i-1
    li.append(l+i-1)
print(sum(li)-remove)

WA
n,l = map(int, input().split())
li = []
for i in range(n):
    li.append(l+i)
target = li[0]
print(li)
for j in range(n):
    if abs(li[j]-0) < abs(li[0]):
        target = li[j]
li.remove(target)
print(sum(li))
print(target)

AC
n,x = map(int, input().split())
l = list(map(int, input().split()))
count = 1
now = 0
for i in range(n):
    now += l[i]
    if now <= x:
        count += 1
    else:
        break
print(count)

132 コードの検証して提出前に直せた
n = int(input())
p = list(map(int, input().split()))
ans = 0
for i in range(1,n-1):
    li = sorted(p[i-1:i+2])
    if li[1] == p[i]:
        ans += 1
print(ans)

133 実装難度C
正直しんどかったが、何とか最後までできた 本当に大きな一勝
添え字は丁寧にやらないと終わり、丁寧にやればいける
ループの時の範囲、添え字の中
n,d = map(int, input().split())
x = [list(map(int, input().split())) for i in range(n)]
ans = 0
for i in range(n-1):#xの何番目のリストが対象なのか
    main = x[i]
    for j in range(i+1,n):#その時に合計で何人の相手と距離を求めるか
        li = []
        for k in range(d):
            num = (abs(main[k]-x[j][k]))**2
            li.append(num)

        target = (sum(li))**(1/2)
        if target.is_integer():
            ans += 1
print(ans)

134
import math
n,d = map(int,input().split())
i = 1 + d
range = (i+d) - (i-d) + 1
print(int(math.ceil(n / range)))

135
n = int(input())
p = list(map(int, input().split()))
a = sorted(p)
count = 0
for i in range(n):
    if p[i] != a[i]:
        count += 1
print("YES" if count <= 2 else "NO")

a
a,b,c = map(int, input().split())
print(c-(a-b) if c > a-b else 0)

136
n = int(input())
ans = 0
for i in range(1,n+1):
    li = list(str(i))
    if len(li)%2:
        ans += 1
print(ans)

137
空白切り出力はリストの要素が文字列でなければならない
k,x = map(int, input().split())
li = [str(i) for i in range(x-k+1,x+k)]
print(" ".join(li))

138
複数の最小公倍数/通分したときの分子
n = int(input())
a = list(map(int, input().split()))
import fractions
from functools import reduce

def lcm_base(x, y):
    return (x * y) // fractions.gcd(x, y)

def lcm(*numbers):
    return reduce(lcm_base, numbers, 1)
lcm = lcm(*a)
li = []
for i in range(n):
    li.append(lcm // a[i])
print(lcm/sum(li))

cf
そのままやったほうが簡単
N = int(input())
A = map(int, input().split())
print('{:.16g}'.format(1 / sum(1 / x for x in A)))


tenka1
a,b,k = map(int, input().split())
li = dict(a = a, b = b, k = k)
now = "a"
for i in range(li["k"]):
    add = 0
    if li[now] % 2:
        li[now] -= 1
    li[now] = int(li[now]/2)
    add += li[now]
    if now == "a":
        now = "b"
    else:
        now = "a"
    li[now] += add
print(li["a"],li["b"])

139
a,b = map(int, input().split())
ans = 0
count = 1
while count < b:
    count += a-1
    ans += 1
print(ans)

140
n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
c = list(map(int, input().split()))

ans = 0

for i in range(n-1):
	if 	a[i+1] == a[i] + 1:
		ans += c[a[i]-1]

print(ans + sum(b))

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
n,k = map(int, input().split())
h = list(map(int, input().split()))
ans = 0
for i in h:
    if i >= k:
        ans += 1
print(ans)
