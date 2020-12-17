class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class Linkedlist():
    def __init__(self):
        self.head = Node(None)
        self.tail = None
    def add(self, data):
        addnode = Node(data)
        if self.head.next == None:
            self.head.next = addnode
            tail = addnode
            addnode.prev = self.head          
        else:
            node = self.head
            while node.next:
                node = node.next
            node.next = addnode
            addnode.prev = node
            tail = addnode
    def delete(self, data):
        if self.head.next == None:
            print("연결리스트에 값이 없습니다.")
            return
        else:
            pre = None
            curnode = self.head
            # head 노드는 data:None, 연결리스트가 있을 경우
            # head.next는 첫번째 연결리스트이다.
            while curnode.next:
                pre = curnode
                curnode = curnode.next
                if(curnode.data == data):
                    temp = curnode
                    pre.next = curnode.next
                    curnode.prev = pre
                    del temp
                    return

    def findindex(self,value):
        if self.head.next == None:
            return False
        else:
            pre = None
            curnode = self.head
            while curnode.next:
                pre = curnode
                curnode = curnode.next
                if(curnode.data > value):
                    return pre.next
            return False
    
    def indexadd(self, index, data):
        if self.head.next == None:
            return
        else:
            temp = None
            curnode = self.head
            addnode = Node(data)
            while curnode.next:
                if(curnode.next == index):
                    temp = curnode.next
                    curnode.next = addnode
                    addnode.next = temp
                    addnode.prev = curnode
                    tail = addnode
                    return addnode.next
    def printlist(self):


T = int(input()) #test_case
for test_case in range(1,T+1):
    N, M = map(int, input().split()) # N: 수열의 길이, M: 수열의 개수
    sequence = Linkedlist()
    for i in range(M):
        tempseq = list(map(int, input().split()))
        if i == 0:
            for j in range(N):
                sequence.add(tempseq[j])
        else:
            index = sequence.findindex(tempseq[0])
            for j in range(N):
                index = sequence.indexadd(index, tempseq[j])