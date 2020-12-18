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
            self.tail = addnode
            addnode.prev = self.head         
        else:
            if self.tail:
                self.tail.next = addnode
                addnode.prev = self.tail
                self.tail = addnode
            else:
                self.head.next = addnode
                addnode.prev = self.head
                self.tail = addnode
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
                    if temp.next:
                        curnode.next = addnode
                        addnode.next = temp
                        temp.prev = addnode
                        addnode.prev = curnode
                    else:
                        curnode.next = addnode
                        addnode.next = temp
                        temp.prev = addnode
                        addnode.prev = curnode
                        self.tail = temp
                    return addnode.next
                curnode = curnode.next
    def printlist(self, result):
        # tail부분에서 프린트
        printnode = self.tail
        count = 0
        while printnode.prev:
            count += 1
            printdata = str(printnode.data)
            result.append(printdata)
            printnode = printnode.prev
            if count>=10: break

T = int(input()) #test_case
for test_case in range(1,T+1):
    N, M = map(int, input().split()) # N: 수열의 길이, M: 수열의 개수
    sequence = Linkedlist()
    result = []
    for i in range(M):
        tempseq = list(map(int, input().split()))
        if i == 0:
            for j in range(N):
                sequence.add(tempseq[j])
        else:
            index = sequence.findindex(tempseq[0])
            if not index:

                for j in range(N):
                    sequence.add(tempseq[j])
            else:
                for j in range(N):
                    index = sequence.indexadd(index, tempseq[j])
    
    sequence.printlist(result)

    print("#{0} {1}".format(test_case ,' '.join(result)))