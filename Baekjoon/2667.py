def ispossible(x, y):
    if (house[x][y] == 1) and (0<=x<N) and (0<=x<N):
        return True
    else:
        return False

def BFS():
    global house
    for i in range(N):
        for j in range(N):
            if house[i][j] == 1:
                cur_x = i
                cur_y = j
                visited[cur_x][cur_y] = 1
                for dir in range(4):
                    next_x = cur_x + dx[dir]
                    next_y = cur_y + dy[dir]
                    if ispossible(next_x, next_y) and not visited[next_x][next_y]:
                        continue

N = int(input())
house = []
for i in range(N):
    house.append(list(map(int, input())))

town = []

dx = [0,1,0,-1] # 상하좌우
dy = [-1,0,1,0]
visited = [[0] * N] * N
BFS()

print(visited)
    
