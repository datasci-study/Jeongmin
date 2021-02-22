# [입력]

# 첫 줄에 테스트케이스의 수 T가 주어진다. 1<=T<=50

# 다음 줄부터 테스트 케이스의 별로 첫 줄에 수열의 길이 N, 추가 횟수 M, 출력할 인덱스 번호 L이 주어지고, 다음 줄에 수열이 주어진다.

# 그 다음 M개의 줄에 걸쳐 추가할 인덱스와 숫자 정보가 주어진다. 5<=N<=1000, 1<=M<=1000, 6<=L<=N+M

# [출력]

# 각 줄마다 "#T" (T는 테스트 케이스 번호)를 출력한 뒤, 답을 출력한다.

T = int(input()) # test_case

for i in range(1, T+1):
    N, M, L = map(int, input().split())

    numlist = list(map(int, input().split()))

    if len(numlist) != N:
        print('처음 지정한 수열의 길이와 맞지 않습니다.')
    
    for j in range(M):
        index, num = map(int, input().split())
        numlist.insert(index, num)
    
    print("#{0} {1}".format(i, numlist[L]))