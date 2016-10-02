from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import pygraphviz as pgv

graph = pgv.AGraph(directed = True)
graph.graph_attr['rankdir'] = 'LR'
graph.graph_attr['splines'] = 'line'
graph.graph_attr['nodesep'] = '.05'
graph.graph_attr['ranksep'] = '1'

inputs = graph.add_subgraph(name = 'inputs')
inputs.graph_attr['label'] = 'Inputs'
for i in range(1, 4):
    inputs.add_node("I" + str(i), color = 'gray', style = 'filled', fillcolor = 'white', shape = 'circle')

hiddens1 = graph.add_subgraph(name = 'hidden')
hiddens1.graph_attr['label'] = 'Hidden Layer 1'
for h1 in range(1, 5):
    hiddens1.add_node("H1" + str(h1), color = 'gray', style = 'filled', fillcolor = 'white', shape = 'circle')

hiddens2 = graph.add_subgraph(name = 'hidden')
hiddens2.graph_attr['label'] = 'Hidden Layer 2'
for h2 in range(1, 5):
    hiddens2.add_node("H2" + str(h2), color = 'gray', style = 'filled', fillcolor = 'white', shape = 'circle')

outputs = graph.add_subgraph(name = 'outputs')
outputs.graph_attr['label'] = 'Outputs'
for o in range(1, 3):
    outputs.add_node("O" + str(o), color = 'gray', style = 'filled', fillcolor = 'white', shape = 'circle')

for i in range(1, 4):
    for h1 in range(1, 5):
        graph.add_edge("I" + str(i), "H1" + str(h1))

for h1 in range(1, 5):
    for h2 in range(1, 5):
        graph.add_edge("H1" + str(h1), "H2" + str(h2))

for h2 in range(1, 5):
    for o in range(1, 3):
        graph.add_edge("H2" + str(h2), "O" + str(o))

graph.draw('droput_before.png', format = 'png', prog = 'dot')