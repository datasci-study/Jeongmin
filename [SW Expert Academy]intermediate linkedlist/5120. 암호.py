
# - 1000이하의 숫자 N개가 주어진다. 이때 시작 숫자가 정해지고, 첫 번째 지정 위치가 된다.

# - 지정 위치부터 M번째 칸을 추가한다. 여기에 앞칸의 숫자와 뒤로 밀려난 칸의 숫자를 더해 넣는다. 추가된 칸이 새로운 지정 위치가 된다. 밀려난 칸이 없으면 시작 숫자와 더한다.

# - 이 작업을 K회 반복하는데, M칸 전에 마지막 숫자에 이르면 남은 칸수는 시작 숫자부터 이어간다.

# - 마지막 숫자부터 역순으로 숫자를 출력하면 비밀번호가 된다. 숫자가 10개 이상인 경우 10개까지만 출력한다.

T = int(input()) #test_case

for i in range(1, T+1):
    N, M, K = map(int, input().split())
    Numlist = list(map(int, input().split()))

    if len(Numlist) != N:
        print('처음 지정한 수열의 길이와 맞지않습니다. ')
    index = 0
    for j in range(K):
        index = index + M
        if index > len(Numlist):
            index = index - len(Numlist)
            
        if index == 0:
            Numlist.insert(index, Numlist[-1]+Numlist[0])
        elif index == len(Numlist):
            Numlist.append(Numlist[-1]+Numlist[0])
        else:
            Numlist.insert(index, Numlist[index-1]+Numlist[index])
    result = reversed(Numlist[-10:])
    print("#{0} ".format(i), end='')
    print(*result)
