T = int(input())

for test_case in range(1,T+1):
    N = int(input())
    mynumber = [0]*N
    mynumber = list(map(int, input().split()))
    mynumber.sort() #오름차순으로 정렬
    result = [0]*N

    bignum = [] 
    smallnum = []
    if (N % 2) == 0: # 짝수개의 인덱스를 갖는 경우
        mid = N - N //2
        
        bignum = mynumber[mid:len(mynumber)] # 중간 기준점을 중심으로 오른쪽은 큰수
        bignum.sort(reverse=True) # 중간기준점을 중심으로 왼쪽은 작은 수
        smallnum = mynumber[0:mid]
        for k in range(0, mid):  
            result[2*k] += bignum[k] #짝수 인덱스에 큰수를 크기 순서대로 차례대로 입력
            result[2*k+1] += smallnum[k] #홀수 인덱수에 작은수를 크기 순서대로 차례대로 입력
    else: # 홀수개의 인덱스를 갖는 경우
        mid = N // 2
        bignum = mynumber[mid:len(mynumber)]
        bignum.sort(reverse=True)
        middle = mynumber[mid] # 중간값이 되는 수
        smallnum = mynumber[0:mid]
        for k in range(1, mid+1):
            result[2*k-2] += bignum[k-1]
            result[2*k-1] += smallnum[k-1]
        result[N-1] += middle # 홀수 인덱스를 가지는 배열에서는 중간값을 갖는 수가 맨 뒤로 입력된다.
    result = result[:10] # 결과값을 10개만 출력하는 경우
    strToresult = ' '.join(map(str, result)) #출력 형식을 맞추기 위해 사용
    print("#{0} {1}".format(test_case, strToresult))
