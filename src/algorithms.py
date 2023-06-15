import math
import collections
from heapq import heappop, heappush
from typing import NewType, List, Dict, Tuple, int
from utils import GraphNode, LinkedList, LinkedListNode, DisjointSet, performQuickSort, heapify, Graph, Edge

class SearchingAlgorithms:
    def linearSearch(A: List[int], x: int) -> int:
        """
        Employs a linear search to find a value x in an array A.
        Returns the first index of x in A else returns None
        """
        for i in range(len(A)):
            if A[i] == x:
                return i
        return None

    def binarySearch(A: List[int], x: int) -> int:
        """
        Employs a binary search to find the insertion point of x in A, a sorted array.
        """
        A.sort()
        l = 0
        r = len(A)
        while l <= r:
            m = (l + r) // 2
            if A[m] == x:
                return m
            elif A[m] > x:
                r = m - 1
            elif A[m] < x:
                l = m + 1
        return r
    
    def breadthFirstSearch(root: GraphNode, v: int) -> GraphNode:
        """
        Employs a breadth first search to find a particular value in an undirected graph with edges.
        Returns None if the value is not found in the graph. 
        """
        visited = set()
        q = collections.deque([root])

        while q:
            node = q.popleft()
            if node.value == v:
                return node
            visited.add(node)

            for edge in node.edges:
                if edge in visited:
                    continue
                q.append(edge)
        return None
    

    def depthFirstSearch(self, root: GraphNode, v: int) -> GraphNode:
        """
        Employs a depth first search to find a particular value in an undirected graph with edges.
        Returns None if the value is not found in the graph. 
        """
        if root.value == v:
            return root
        for edge in root.edges:
            result = self.depthFirstSearch(edge, v)
            if result is not None:
                return result
        return None

    def kruskalMST(edges: List[Edge], num_vertices: int) -> List[Edge]:
        """
        Uses the Kruskal MST implementation
        """
        # Sort the edges in ascending order based on weight
        edges.sort()

        disjoint_set = DisjointSet(num_vertices)
        mst = []

        for edge in edges:
            weight, src, dest = edge
            if disjoint_set.find(src) != disjoint_set.find(dest):
                disjoint_set.union(src, dest)
                mst.append(edge)
        return mst

    def primMST(edges: List[Edge], num_vertices: int) -> List[Edge]:
        """
        Uses the Prim MST implementation
        """
        graph = collections.defaultdict(list)

        for weight, src, dest in edges:
            graph[src].append((weight, dest))
            graph[dest].append((weight, src))

        mst = []
        visited = set()

        start_vertex = 0

        visited.add(start_vertex)

        while len(visited) < num_vertices: # While the vertices haven't been exhausted
            min_edge = None
            # Find the minimum-weight edge with one end in the visited set and the other end not in the visited set
            for u in visited:
                for weight, v in graph[u]:
                    if v not in visited and (min_edge is None or weight < min_edge[0]):
                        min_edge = (weight, u, v)
            
            if min_edge is None:
                break

            weight, u, v = min_edge
            mst.append(min_edge)
            visited.add(v)

        return mst


    
    def dijkstraShortestPath(graph: Graph, start_vertex: int) -> Dict[int, int]:
        """
        Implements Dijkstra's shortest path algorithm to keep a dictionary which stores the shortest 
        path from the start vertex to a given vertex in a graph with weighted edges
        """
        distances = {vertex : float('inf') for vertex in graph}
        distances[start_vertex] = 0

        pq = [(0, start_vertex)]

        while pq:
            curr_distance, curr_vertex = heappop(pq)

            if curr_distance > distances[curr_vertex]:
                continue

            for neighbor_dist, neighbor_vert in graph[curr_vertex]:
                distance = curr_distance + neighbor_dist

                if distance < distances[neighbor_vert]:
                    distances[neighbor_vert] = distance
                    heappush(pq, (distance, neighbor_vert))

        return distances


class SortingAlgorithms:
    def bubbleSort(A: List[int]) -> List[int]:
        """
        Implements bubble sort to return a sorted array.
        """
        n = len(A)
        for i in range(n - 1):
            for j in range(i, n - 1):
                if A[j] > A[j + 1]:
                    A[j], A[j + 1] = A[j + 1], A[j]
        return A
    
    def bucketSort(self, A: List[int]) -> List[int]:
        """
        Implements bucket sort to return a sorted array.
        """
        n = len(A)
        res = []
        buckets = [LinkedList() for _ in range(n)]

        for i in range(n):
            buckets[math.floor(n*A[i])].insert(A[i])
        
        for i in range(n):
            buckets[i].insertion_sort()

        for head in buckets:
            node = head
            while node:
                res.append(node.value)
                node = node.next
        return res
    
    def insertionSort(A: List[int]) -> List[int]:
        """
        Implements insertion sort to return a sorted array.
        """
        n = len(A)
        for i in range(1, n):
            key = A[i]
            j = i - 1
            while j >= 0 and A[j] > key:
                A[j+1] = A[j]
                j -= 1
            A[j+1] = key
        return A

    def selectionSort(A: List[int]) -> List[int]:
        """
        Implements selection sort to return a sorted array.
        """
        n = len(A)
        for i in range(n):
            min_index = i
            for j in range(i+1, n):
                if A[j] < A[min_index]:
                    min_index = j
            A[i], A[min_index] = A[min_index], A[i]
        return A
    
    def quicksort(A: List[int]) -> List[int]:
        """
        Implements quicksort to return a sorted array.
        See utils for algorithm implementation 
        """
        n = len(A)
        performQuickSort(A, 0, n)
        return A

    def radixSort(A: List[int]) -> List[int]:
        pass

    def heapSort(A: List[int]) -> List[int]:
        n = len(A)

        for i in range(n//2, -1, -1):
            heapify(A, n, i)
        
        for i in range(n-1, 0, -1):
            # Swap max heap with last element
            A[i], A[0] = A[0], A[i]
            # Heapify after reducing n by 1 and displacing the max element
            heapify(A, i, 0)
        
        return A



