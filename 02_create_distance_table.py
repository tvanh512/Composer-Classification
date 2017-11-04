import numpy as np
from utils import encode, get_power_set, Vertex, Graph, shortest, dijkstra,neighbor_near, neighbor_far
from copy import copy
import sys
import pickle
import config

# Create named chords

named_chords_encode = pickle.load(open(config.named_chords_encode_path,"rb" ))

# Create a space of 12^4 = 4096 possible chord subset 
chord_sets = get_power_set([0,1,2,3,4,5,6,7,8,9,10,11])
chord_sets.remove([])

# Create neighbor set, neighbor near have distance w1, neighbor far have distance w2
neighbor_near_list ={}
neighbor_far_list ={}
for i in range(len(chord_sets)):  
    neighbor_near_list.update(neighbor_near(chord_sets[i]))
for i in range(len(chord_sets)):  
    neighbor_far_list.update(neighbor_far(chord_sets[i]))

# Create distance table and save it to file. 
# Later, we can look up the table to find distance between 2 chords
w1 = 1
w2 = 3
weight_table =[]
count = 0
from datetime import datetime
for chord in named_chords_encode:
    print str(datetime.now())
    print count
    g = Graph()
    for i in range(1,4096):
        g.add_vertex(i)
    for i in range(1,4096):
        for j in neighbor_near_list[i]:
            g.add_edge(i,j,w1)
        for k in neighbor_far_list[i]:
            g.add_edge(i,k,w2)
    dijkstra(g,g.get_vertex(chord))
    for i in range(1,4096):
        n = g.get_vertex(i).get_distance()
        weight_table.append([chord,i,n])
    count = count + 1

pickle.dump(distance_table,open(config.distance_table, "wb" ) )
