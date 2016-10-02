from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import pygraphviz as pgv

graph = pgv.AGraph(directed = True)
graph.graph_attr['rankdir'] = 'TB'
graph.graph_attr['splines'] = 'line'
graph.graph_attr['nodesep'] = '.05'
graph.graph_attr['ranksep'] = '1'

inputs = graph.add_subgraph(name = 'inputs')
inputs.graph_attr['label'] = 'Inputs'
for i in range(1, 9):
    inputs.add_node("R" + str(i), color = 'red', style = 'filled', fillcolor = 'red', shape = 'circle')
    inputs.add_node("G" + str(i), color='green', style = 'filled', fillcolor = 'green', shape = 'circle')
    inputs.add_node("B" + str(i), color='blue', style = 'filled', fillcolor = 'blue', shape = 'circle')

hiddens = graph.add_subgraph(name = 'hidden')
hiddens.graph_attr['label'] = 'Hidden Layer'
for h in range(1, 20):
    hiddens.add_node("H" + str(h), color = 'gray', style = 'filled', fillcolor = 'white', shape = 'circle')

outputs = graph.add_subgraph(name = 'outputs')
outputs.graph_attr['label'] = 'Outputs'
for o in range(1, 4):
    outputs.add_node("O" + str(o), color = 'gray', style = 'filled', fillcolor = 'white', shape = 'circle')

for i in range(1, 9):
    for h in range(1, 20):
        graph.add_edge("R" + str(i), "H" + str(h))
        graph.add_edge("G" + str(i), "H" + str(h))
        graph.add_edge("B" + str(i), "H" + str(h))

for h in range(1, 20):
    for o in range(1, 4):
        graph.add_edge("H" + str(h), "O" + str(o))

graph.draw('image_nn.png', format = 'png', prog = 'dot')