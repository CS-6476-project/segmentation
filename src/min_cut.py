from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import random


def contract(g, v, w):
    wNeighbors = g.neighbors(w)
    wNeighbors = list(wNeighbors)
    for node in wNeighbors:
        if node != v:
            g.add_edge(v, node)
        g.remove_edge(node, w)
        if node != v:
            g.add_edge(node, v)
    wNeighbors = g.neighbors(w)
    wNeighbors = list(wNeighbors)
    if g.nodes[v]['labels'] == None:
        g.nodes[v]['labels'] = []
    g.nodes[v]['labels'] = g.nodes[v]['labels'] + (g.nodes[w]['labels'])
    g.remove_node(w)


img = data.coffee()

labels1 = segmentation.slic(img, compactness=30, n_segments=400)

g = graph.rag_mean_color(img, labels1, mode='similarity')

while len(g) > 10:
    v = random.choice(list(g.nodes))
    # print(v)
    vNeighbors = g.neighbors(v)
    list1 = list(vNeighbors)
    # print(list1)
    w = random.choice((list1))
    # print(w)
    contract(g, v, w)

print(g.nodes)
labels = [[] for i in range(10)]
counter = 0
for node in g.nodes:
    print(g.nodes[node])
    labels[counter] = labels[counter] + g.nodes[node]['labels']
    counter = counter + 1
print(labels)

labels2 = labels1.copy()
print(labels2.shape)
for x in range(labels2.shape[0]):
    for y in range(labels2.shape[1]):
        old = labels2[x][y]
        for z in range(len(labels)):
            if old in labels[z]:
                labels2[x][y] = labels[z][0]


# print(labels2)

out2 = color.label2rgb(labels2, img, kind='avg')

plt.imshow(labels2)
plt.show()
