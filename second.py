n, q = map(int , input().split(" "))
a = dict()
pre = -1
max = 0
sum = 1
for i in range(q):
    h = input().split(' ')
    if a.get(int(h[0])) != None:
        if int(h[1]) > a.get(int(h[0])):
            a[int(h[0])] = int(h[1])
    else:
        a[int(h[0])] = int(h[1])
    x = int(h[1]) - int(h[0]) + 1
    if x > max:
        max = x
for key in sorted(a.keys()):
    x = a[key] - key + 1
    counter = 0
    j = 0
    if pre >= key:
        counter = pre - key + 1
        j = j + counter
    while j < x:
        sum = (sum *(max - counter)) % 1000000007
        counter += 1
        n-= 1
        j += 1
    if pre < a[key]:
        pre = a[key]
for i in range(n):
    sum = (sum * max) % 1000000007
print(str(max)+" "+str(sum % 1000000007))