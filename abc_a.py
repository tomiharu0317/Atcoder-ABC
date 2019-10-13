# 01
h1 = int(input())
h2 = int(input())
print(h1 - h2)

# 02
x,y = map(int, input().split())
print(max(x,y))

# 03
n = int(input())
def total(n):
    if n < 1:
        return 0
    return n + total(n-1)
print((10000 * total(n))/n)
#初再帰関数

# 04
print(int(input())*2)

# ここからずれている
# 04
n = int(input())
data = [map(int,input().split()) for i in range(n)]
flat = [flatten for inner in data for flatten in inner]
x = flat[::2]
y = flat[1::2]
ans = []
for i in range(n-1):
    for j in range(n-1):
        dist = (abs((x[i] - x[j+1]))**2 + abs((y[i] - y[j+1]))**2)**(1/2)
        ans.append(dist)
print(max(ans))

# 05
x,y = map(int, input().split())
print(int(y//x))

# 06
n = input()
if "3" in n or int(n)%3==0:
    print("YES")
else:
    print("NO")

# 07
print(int(int(input()) - 1))

# 08
s,t = map(int, input().split())
print(int(t-s+1))

# 09
import math
print(int(math.ceil(int(input())/2)))

# 10
print(input()+"pp")

# 11
n = int(input())
if n == 12:
    print(1)
else:
    print(n+1)

# 12
a,b = map(int, input().split())
print(b,a)

# 13
li = ["A","B","C","D","E"]
x = input()
print(li.index(x) + 1)

# 14
a = int(input())
b = int(input())
if a%b:
    print((((a//b)+1)*b)-a)
else:
    print(0)

# 15
a = input()
b = input()
if len(a) <= len(b):
    print(b)
else:
    print(a)

# 16
m,d = map(int,input().split())
print("YES" if m%d == 0 else "NO" )

# 17
s1,e1 = map(int,input().split())
s2,e2 = map(int,input().split())
s3,e3 = map(int,input().split())
print(int(s1 *(e1/10) + s2 *(e2/10) + s3 *(e3/10)))

# 18
a = int(input())
b = int(input())
c = int(input())
print(1 if a == max(a,b,c) else 3 if a == min(a,b,c) else 2)
print(1 if b == max(a,b,c) else 3 if b == min(a,b,c) else 2)
print(1 if c == max(a,b,c) else 3 if c == min(a,b,c) else 2)

# cf index をかえすのがよい
A = int(input())
B = int(input())
C = int(input())

original = [A,B,C]
ranking = sorted(original)
ranking.reverse()
for i in range(len(original)):
    print(ranking.index(original[i])+1)

# 19
li = sorted(list(map(int, input().split())))
print(li[1])

# 20
q = int(input())
print("ABC" if q == 1 else "chokudai")

# 21
n = int(input())
print(n)
for i in range(n):
    print(1)

# 22
n,s,t = map(int, input().split())
w = int(input())
li = [int(input()) for i in range(n-1)]
ans = 0
if s <= w <= t:
    ans = 1
for i in range(n-1):
    w += li[i]
    if s <= w <= t:
        ans += 1
print(ans)

# 23
x = input()
ans = 0
for i in range(len(x)):
    ans += int(x[i])
print(ans)

# 24
a,b,c,k = map(int, input().split())
s,t = map(int, input().split())
ans = 0
ans = s*a + t*b
if s+t >= k:
    ans -= c*(s+t)
print(ans)

# 25
li = sorted(list(input()))
n = int(input())
dict = []
for i in range(5):
    for j in range(5):
        dict.append(li[i]+li[j])
print(dict[n-1])

# 26
n = int(input())
if n % 2:
    n -= 1
    a = n/2
    print(int(a*(a+1)))
else:
    print(int((n/2)**2))

# 27
li = sorted(list(map(int, input().split())))
if li[0] == li[1]:
    print(li[2])
else:
    print(li[0])

# 28
n = int(input())
if n <= 59:
    print("Bad")
elif 60 <= n <= 89:
    print("Good")
elif 90 <= n <= 99:
    print("Great")
else:
    print("Perfect")

# 29
print(input()+"s")

# 30
a,b,c,d = map(int, input().split())
taka = b/a
aoki = d/c
if taka > aoki:
    print("TAKAHASHI")
elif taka < aoki:
    print("AOKI")
else:
    print("DRAW")

# 31
a,d = map(int, input().split())

if (a+1)*d >= a*(d+1):
    print((a+1)*d)
else:
    print(a*(d+1))

# 32☆
a = int(input())
b = int(input())
n = int(input())
def gcd(a,b):
    while b:
        a,b = b, a % b
    return a
lcm = a * b // gcd(a,b)
while n % lcm != 0:
    n += 1
else:
    print(n)

# 33
li = list(input())
n = li.count(li[0])
if n == 4:
    print("SAME")
else:
    print("DIFFERENT")

# 34
x,y = map(int,input().split())
if x < y:
    print("Better")
else:
    print("Worse")

# 35
w,h = map(int, input().split())
if w/h == 4/3:
    print("4:3")
else:
    print("16:9")

# 36
a,b = map(int, input().split())
n = b // a
if a * n == b:
    print(n)
else:
    print(n+1)

# 37
a,b,c = map(int, input().split())
if a >= b:
    print(c//b)
else:
    print(c//a)

# 38
print("YES" if input()[-1] == "T" else "NO")

# 39
a,b,c = map(int, input().split())
print(a*b*2 + b*c*2 + c*a*2)

# 40
n, x = map(int, input().split())
print(min(x-1, n-x))

# 41
s = input()
i = int(input())
print(s[i-1])

42
if input().split().count('5') == 2:
    print('YES')
else:
    print('NO')

43
n = int(input())
print(int(n*(n+1)/2))

44
n = int(input())
k = int(input())
x = int(input())
y = int(input())

if n > k:
    print(int(x*k + y*(n-k)))
else:
    print(int(x*n))

45
a = int(input())
b = int(input())
h = int(input())
print(int((a+b)*h/2))

46
li = list(map(int, input().split()))
li = set(li)
print(len(li))

47
li = sorted(list(map(int, input().split())))
if li[0] == sum(li[1:3]) or sum(li[0:2]) == li[2]:
    print("Yes")
else:
    print("No")

48
a,b,c = map(str, input().split())
b = list(b)[0]
print("A"+str(b)+"C")

49
li = ["a","e","i","o","u"]

if input() in li:
    print('vowel')
else:
    print('consonant')

50
a,op,b = map(str, input().split())
a = int(a)
b = int(b)

if op == "+":
    print(int(a + b))
if op == "-":
    print(int(a - b))

51
print(input().replace(",", " "))

52
a,b,c,d = map(int, input().split())

if a*b <= c*d:
    print(c*d)
if a*b > c*d:
    print(a*b)

53
if int(input()) < 1200:
    print("ABC")
else:
    print("ARC")

54
a, b = map(int, input().split())
if a == 1 and a != b:
    print("Alice")
    exit()
if b == 1 and a != b:
    print("Bob")
    exit()
if a < b:
    print("Bob")
    exit()
if a > b:
    print("Alice")
    exit()
if a == b:
    print("Draw")

55
n = int(input())
print(800*n - (n//15)*200)
#//で整数の商

56
a, b = map(str, input().split())

if a == b:
    print("H")
else:
    print("D")

57
a,b = map(int, input().split())

if a+b < 24:
    print(a+b)
if a+b >= 24:
    print(a+b-24)

58
a,b,c = map(int, input().split())
if b-a == c-b:
    print("YES")
else:
    print("NO")

59
a,b,c = map(str, input().split())
print((a[0] + b[0] + c[0]).upper())

60
a,b,c = map(str, input().split())
if a[-1] == b[0] and b[-1] == c[0]:
    print("YES")
else:
    print("NO")

61
a,b,c = map(int, input().split())
if a <= c <= b:
    print("Yes")
else:
    print("No")

62
a = [1,3,5,7,8,10,12]
b = [4,6,9,11]
c = [2]
x,y = map(int, input().split())
if (x in a and y in a) or (x in b and y in b) or (x in c and y in c):
    print("Yes")
else:
    print("No")

63
a,b = map(int, input().split())
if a+b >= 10:
    print("error")
else:
    print(a+b)

64
r,g,b = map(str, input().split())
rgb = int(r+g+b)
if rgb % 4:
    print("NO")
else:
    print("YES")

65
x,a,b = map(int, input().split())

if a >= b:
    print("delicious")
if a < b and abs(a-b) <= x:
    print("safe")
if a < b and abs(a-b) > x:
    print("dangerous")

66
li = sorted(list(map(int, input().split())))
print(sum(li[0:2]))

67
a,b = map(int, input().split())
if a % 3 ==0 or b % 3 == 0 or (a+b) % 3 == 0 :
    print("Possible")
else:
    print("Impossible")

68
print("ABC"+input())

69☆
n,m = map(int, input().split())
print((n-1)*(m-1))

70
li = list(input())
if li[0] == li[2]:
    print("Yes")
else:
    print("No")

71
x,a,b = map(int, input().split())
if abs(x-a)>abs(x-b):
    print("B")
else:
    print("A")

72
x,t = map(int, input().split())
if x >= t:
    print(x-t)
if x < t:
    print(0)

73
li = list(input())
if "9" in li:
    print("Yes")
else:
    print("No")

74
n = int(input())
a = int(input())
print(n*n - a)

75☆
li = list(map(int, input().split()))
if li.count(li[0]) == 1:
    print(li[0])
if li.count(li[1]) == 1:
    print(li[1])
if li.count(li[2]) == 1:
    print(li[2])

20190727
li = sorted(map(int, input().split()))
print(li[0] if li[1]==li[2] else li[2])

76
r = int(input())
g = int(input())
print(2*g-r)

77
a = list(input())
b = list(input())[::-1]
if a == b:
    print("YES")
else:
    print("NO")

78
li = list("ABCDEF")
x, y = map(str, input().split())
if li.index(x) < li.index(y):
    print("<")
if li.index(x) > li.index(y):
    print(">")
if li.index(x) == li.index(y):
    print("=")
#index()

79
li = list(input())
if (li[0]==li[1]==li[2])or(li[1]==li[2]==li[3]):
    print("Yes")
else:
    print("No")

80
n,a,b = map(int, input().split())

if a*n >= b:
    print(b)
    exit()
if a*n <= b:
    print(a*n)

81
li = list(input())
print(li.count("1"))

82
import math
a,b = map(int, input().split())
print(math.ceil((a+b)/2))
#切り上げmath.ceil() 切り捨てmath.floor()

83
a,b,c,d = map(int, input().split())
if a+b>c+d:
    print("Left")
if a+b<c+d:
    print("Right")
if a+b == c+d:
    print("Balanced")

84
print(24+(24-int(input())))

85
print(input().replace("2017", "2018"))

86
a,b = map(int, input().split())
if a*b % 2:
    print("Odd")
else:
    print("Even")

87
x = int(input())
a = int(input())
b = int(input())
c = (x-a)//b
print(x-a-(b*c))

88☆
n = int(input())
a = int(input())
if n<a:
    print("Yes")
    exit()
if n % 500 <= a:
    print("Yes")
else:
    print("No")

20190727
n = int(input())
a = int(input())
print("Yes" if n % 500 <= a else "No")

89
print(int(input())//3)

90
a = list(input())
b = list(input())
c = list(input())
print(a[0]+b[1]+c[2])

91
a,b,c = map(int, input().split())
if a+b >= c:
    print("Yes")
if a+b < c:
    print("No")

92
a = int(input())
b = int(input())
c = int(input())
d = int(input())
ans = 0

if a<=b:
    ans += a
else:
    ans += b
if c<=d:
    ans += c
else:
    ans += d
print(ans)

93
s = list(input())
if "a" in s and "b" in s and "c" in s:
    print("Yes")
else:
    print("No")

94
a,b,x = map(int, input().split())
if a <= x <= a+b:
    print("YES")
else:
    print("NO")

95
print(700 + 100* input().count("o"))

96
a,b = map(int, input().split())
if a > b:
    print(a-1)
if a <= b:
    print(a)

97
a,b,c,d = map(int, input().split())
if (abs(b-a)<=d and abs(c-b)<=d) or abs(c-a)<=d:
    print("Yes")
else:
    print("No")

98
a,b = map(int,input().split())
print(max(a+b,a-b,a*b))

99
if 1<= int(input()) <= 999:
    print("ABC")
else:
    print("ABD")

100
a,b = map(int, input().split())
if a <= 8 and b <= 8:
    print("Yay!")
else:
    print(":(")

101
li = list(input())
print(1*li.count("+") + -1*li.count("-"))

102
n = int(input())
if n % 2:
    print(2*n)
else:
    print(n)

103
li = sorted(list(map(int, input().split())))
print(abs(li[0] - li[1]) + abs(li[1] - li[2]))

104
r = int(input())
if r < 1200:
    print("ABC")
if 1200 <= r < 2800:
    print("ARC")
if 2800 <= r:
    print("AGC")

105
n, k = map(int, input().split())
if n % k:
    print(1)
else:
    print(0)

106
a,b = map(int, input().split())
print((a-1)*(b-1))

107
n,i = map(int, input().split())
print(n-i+1)

108
k = int(input())
if k % 2 :
    print(int(((k-1)/2) * ((k+1)/2)))
else:
    print(int((k/2) * (k/2)))

109
a,b = map(int, input().split())
if a*b % 2:
    print("Yes")
else:
    print("No")

110
li = sorted(list(map(int, input().split())),reverse = True)
print(int(str(li[0])+str(li[1]))+li[2])

111
print(input().replace("1","8").replace("9","1").replace("8","9"))

112
n = int(input())
if n == 1:
    print("Hello World")
if n == 2:
    a = int(input())
    b = int(input())
    print(a+b)

113
x,y = map(int, input().split())
print(int(x+(y/2)))

114
n = int(input())
if n==7 or n==5 or n==3:
    print("YES")
else:
    print("NO")

115
d = int(input())
if d == 25:
    print("Christmas")
if d == 24:
    print("Christmas Eve")
if d == 23:
    print("Christmas Eve Eve")
if d == 22:
    print("Christmas Eve Eve Eve")

116
ab,bc,ca = map(int, input().split())
print(int(ab * bc * 1/2))


117
t, x = map(int, input().split())
print(t/x)

118
a,b = map(int,input().split())
if b%a :
    print(b-a)
else:
    print(a+b)

119
a = ["01", "02", "03", "04"]
li = list(input().split("/"))
if li[1] in a:
    print("Heisei")
else:
    print("TBD")

120
a,b,c = map(int, input().split())
if b // a >= c:
    print(c)
if b // a < c:
    print(b // a)

121☆
uh, uw = map(int, input().split())
h, w = map(int, input().split())
print(int((uh*uw) - (uh*w) - (uw*h) + (h*w)))

20190727
uh,uw = map(int, input().split())
h, w = map(int, input().split())
print((uh-h)*(uw-w))

122
b = input()
if b == "A":
    print("T")
if b == "T":
    print("A")
if b == "G":
    print("C")
if b == "C":
    print("G")

123☆
a = int(input())
b = int(input())
c = int(input())
d = int(input())
e = int(input())
k = int(input())
if e - a > k:
    print(":(")
    exit()
print("Yay!")

20190727
li = [int(input()) for i in range(6)]
print("Yay!" if li[4]-li[0] <= li[5] else ":(")

124
li = sorted(list(map(int, input().split())))
if li[1] - 1 >= li[0]:
    print(int(2 * li[1] - 1))
else:
    print(sum(li))

125
a,b,t = map(int, input().split())
print(int(b * (t // a)))

126☆☆
n, k = map(int, input().split())
s = input()
print(s[:k-1] + s[k-1].lower() + s[k:])

replace()第三引数で回数指定できる

cf.
n,k = map(int, input().split())
s = input()
print(s.replace(s[k-1], s[k-1].lower()))
#これだと文字が重複したときにすべて変換される

127
a,b = map(int, input().split())
if a <= 5:
    print(0)
if 6 <= a <= 12:
    print(int(b/2))
if 13 <= a:
    print(b)

128
a,p = map(int, input().split())
print(int((3*a + p)//2 ))

129
li = sorted(list(map(int, input().split())))
print(li[0] + li[1])

130
x,a = map(int, input().split())
if x < a:
    print(0)
if x >= a:
    print(10)

131
s = input()
print("Bad" if s[0]==s[1] or s[1]==s[2] or s[2]==s[3] else "Good")

132
s = sorted(list(input()))
print("Yes" if s.count(s[0])==2 and s.count(s[2])==2 else "No")

133
n,a,b = map(int, input().split())
print(n*a if n*a <= b else b)

134
print(3*(int(input())**2))

135
a,b = map(int, input().split())
if (a+b)%2 == 0:
    print(int((a+b)/2))
else:
    print("IMPOSSIBLE")

136
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

137
a,b = map(int, input().split())
print(max(a+b,a-b,a*b))

138
a = int(input())
s = input()
print(s if a >= 3200 else "red")

tenka1
s = input()
print(s if len(s) == 2 else s[::-1])

139
s = input()
t = input()
count = 0
for i in range(3):
    if s[i] == t[i]:
        count += 1
print(count)

140
s = input()

if s == "Sunny":
    print("Cloudy")

elif s == "Cloudy":
    print("Rainy")

else:
    print("Sunny")

141
n = int(input())
print(n**3)

142
n = int(input())
if n % 2:
    print(-(-n//2) / n)
else:
    print((n//2) / n)
