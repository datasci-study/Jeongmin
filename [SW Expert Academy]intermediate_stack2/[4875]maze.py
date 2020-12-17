
def issafe(y,x):
    return 0 <= x < N and 0 <= y < N and (maze[y][x] == 0 or maze[y][x] == 3)

def FindRoutes(y,x):
    result = 0
    if maze[y][x]==3:
        result = 1
        return result
    visited.append((x, y))
    for dir in range(4):
        New_x = x + dx[dir]
        New_y = y + dy[dir]
        if issafe(New_y, New_x) and (New_y, New_x) not in visited:
            FindRoutes(New_y, New_x)

    
T = int(input())
for test_case in range(1,T+1):
    N = int(input()) # 미로의 크기 N
    maze = []
    maze_arr = [] # 미로의 통로와 벽에 대한 정보
    visited = [] # 방문한 미로의 좌표값
    result = 0

    # 상, 하, 좌, 우
    dx = [0,0,-1,1]
    dy = [-1,1,0,0]
    for j in range(N):
        maze_arr = (list(map(int, input())))
        maze.append(maze_arr)
    for y in range(N):
        for x in range(N):
            if(maze[y][x]==2):
                y_start = y
                x_start = x
    result = FindRoutes(y_start, x_start)

    print("#{}: {}".format(test_case,result))
