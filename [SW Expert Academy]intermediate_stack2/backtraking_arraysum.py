
def find(row):
    global subsum, result

    # 기존 결과가 더 작다면 그냥 return
    if result < subsum:
        return

    # 마지막 row에 도달한 경우
    if row == size:
        # 기존 결과보다 작다면 result 갱신
        if subsum < result:
            result = subsum
        # 아니면 그냥 return
        return

    # Unvisited col.에 대해 iteration
    for col in range(size):
        if not visit[col]:
            # 해당 col. 방문 후 subsum 증가
            visit[col] = 1
            subsum += board[row][col]
            # 다음 row로 이동, 재귀적으로 마지막 row까지 호출
            find(row+1)
            # 호출 stack에서 함수가 꺼내지면서 visit과 subsum도 초기화
            visit[col] = 0
            subsum -= board[row][col]


T = int(input())
for test_case in range(1, T + 1):
    size = int(input())
    board = [list(map(int, input().split())) for _ in range(size)]

    visit = [0]*size
    subsum = 0
    result = 2 ^ 31 # Some big number

    find(0) # Start from row 0

    print('#%d %d' % (test_case, result))