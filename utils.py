import numpy as np
from copy import copy
import sys 

def encode(x):
    out = 0
    for i in range(len(x)):
        out = out + 2**(x[i])
    return out

def get_power_set(s):
    power_set=[[]]
    for elem in s:
        # iterate over the subsets
        for sub_set in power_set:
            # add a new subset
            power_set=power_set+[list(sub_set)+[elem]]
    return power_set


# Create neigher with distance w1
def neighbor_near(C):
    fneg = []  #fneg is one element changed by -1
    fpov = []  #fneg is one element changed by +1
    fneg_encode =[]
    fpov_encode = []
    m = len(C)
    for i in range(m):
        fneg.append(copy(C))
        fneg[i][i] = (fneg[i][i] - 1) % 12
        fpov.append(copy(C))
        fpov[i][i] = (fpov[i][i] + 1) % 12
    for i in range(m):
        fneg[i] = list(np.unique(fneg[i]))
        fpov[i] = list(np.unique(fpov[i]))
    for element in fneg:
        if len(element) < m:
            fneg.remove(element)
    for element in fpov:
        if len(element) < m:
            fpov.remove(element)
    
    for i in range(len(fneg)):
        fneg_encode.append(encode(fneg[i]))
    for i in range(len(fpov)):
        fpov_encode.append(encode(fpov[i]))    
    return {encode(C):(fneg_encode+fpov_encode)}

# Create neighbor with distance w2
def neighbor_far(C):
    gneg = []  #one element delete
    gpov = []  #one element add
    m = len(C)
    
    if m >=2:
        for i in range(m):
            n = copy(C)
            n.remove(n[i])
            gneg.append(encode(n))

    if m < 12:
        for j in range(0,12):
            if j not in (k for k in C):
                t = copy(C)
                t.append(j)                
                gpov.append(encode(t)) 
    return {encode(C):(gneg+gpov)}

class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        # Set distance to infinity for all nodes
        self.distance = sys.maxint
        # Mark all nodes unvisited        
        self.visited = False  
        # Predecessor
        self.previous = None

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous

def shortest(v, path):
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return

import heapq

def dijkstra(aGraph, start):
    #print '''Dijkstra's shortest path'''
    # Set the distance for the start node to zero 
    start.set_distance(0)

    # Put tuple pair into the priority queue
    unvisited_queue = [(v.get_distance(),v) for v in aGraph]    
    heapq.heapify(unvisited_queue)
    while len(unvisited_queue):
        # Pops a vertex with the smallest distance 
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        #for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next)
            
            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
         
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        unvisited_queue = [(v.get_distance(),v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)  
