n = int(input())
data = []
for i in range(n):
    data.append(int(input()))
sum = 0
for i in range(n):
    try:
        sum = sum + ((pow(data[i], 2))/(i-1))
        print(sum)
    except ZeroDivisionError:
        print("Zero Error")