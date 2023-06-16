import collections
from heapq import heappop, heappush
from typing import NewType, List, Dict, Tuple, int

class GraphNode:
    def __init__(self, value):
        self.value = value
        self.edges = []

# Edge is used for Kriskal and Prim MST
Edge = tuple[None, None, None]

Graph = Dict[int, List[Tuple[int, int]]]

class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

class LinkedListNode:
    def __init__(self, value=None):
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert(self, v: int):
        new = LinkedListNode(v)
        if self.head is None:
            self.head = new
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = new
    
    def insertion_sort(self):
        if self.head is None or self.head.next is None:
            return
        
        sorted_list = None
        current = self.head

        while current is not None:
            next = current.next
            sorted_list = self.sorted_insert(sorted_list, current)
            current = next
        
        self.head = sorted_list
    
    def sorted_insert(self, sorted_list, new_node: LinkedListNode):
        if sorted_list is None or new_node.value < sorted_list.value:
            new_node.next = sorted_list
            return new_node
        
        current = sorted_list
        while current.next is not None and current.next.value < new_node.value:
            current = current.next

        new_node.next = current.next
        current.next = new_node
        return sorted_list
    
def partition(A, low, hi):
    pivot = A[hi] # choose the pivot at the rightmost location
    i = low - 1
    for j in range(low, hi):
        if A[j] <= pivot:
            i = i + 1
            A[i], A[j] = A[j], A[i]
    A[i + 1], A[hi] = A[hi], A[i + 1] 
    return i + 1

def performQuickSort(A, low, hi):
    if low < hi:
        pi = partition(A, low, hi)

        # recurse on left partition
        performQuickSort(A, low, pi - 1)
        # recurse on right partition
        performQuickSort(A, pi + 1, hi)

def heapify(A, n, i):
    largest = i
    l = i * 2 + 1
    r = i * 2 + 2

    if l < n and A[l] > A[largest]:
        largest = l
    
    if r < n and A[r] > A[largest]:
        largest = r
    
    if largest != i:
        A[i], A[largest] = A[largest], A[i]
        heapify(A, n, largest)


def countingSort(A: List[int], place: int) -> List[int]:
    size = len(A)
    output = [0] * size
    count = [0] * 10

    # find the count of elements
    for i in range(0, size):
        index = A[i] // place
        count[index % 10] += 1

    # Find cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]

    # place the elements in sorted order
    i = size - 1
    while i >= 0:
        index = A[i] // place
        output[count[index % 10] - 1] = A[i]
        count[index % 10] -= 1
        i -= 1
    
    for i in range(0, size):
        A[i] = output[i]

