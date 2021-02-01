import netgraph
import pandas as pd
import spacy
import networkx as nx  # a really useful network analysis library
import matplotlib.pyplot as plt
from math import sqrt
from networkx.algorithms import community  # not used, yet...
import datetime  # access to %%time, for timing individual notebook cells
import os
import holoviews as hv
from holoviews import opts
from bokeh.plotting import show
import numpy as np

# A more detailed model (with higher-dimension word vectors) - 13s to load, normally
nlp = spacy.load(
    r'C:\Users\salma\PycharmProjects\qual_analysis_m12\.env\Lib\site-packages\en_core_web_lg\en_core_web_lg-2.2.5')

# a smaller model, e.g. for testing
# nlp = spacy.load('en_core_web_md')

plt.rcParams['figure.figsize'] = [50, 50]  # makes the output plots large enough to be useful

data = pd.read_csv('complete_dataset.csv')
data.shape
data.head(6)

tokens = []
lemma = []
pos = []
parsed_doc = []
col_to_parse = 'col1'

for doc in nlp.pipe(data[col_to_parse].astype('unicode').values, batch_size=50, n_threads=3):
    if doc.is_parsed:
        parsed_doc.append(doc)
        tokens.append([n.text for n in doc])
        lemma.append([n.lemma_ for n in doc])
        pos.append([n.pos_ for n in doc])
    else:
        # We want to make sure that the lists of parsed results have the
        # same number of entries of the original Dataframe, so add some blanks in case the parse fails
        tokens.append(None)
        lemma.append(None)
        pos.append(None)

data['parsed_doc'] = parsed_doc
data['comment_tokens'] = tokens
data['comment_lemma'] = lemma
data['pos_pos'] = pos

data.head(6)

stop_words = spacy.lang.en.stop_words.STOP_WORDS
print("\n")
print('Number of stopwords: %d' % len(stop_words))
print(list(stop_words))

world_data = data

# takes 1s for 500 nodes - but of course this won't scale linearly!
raw_G = nx.Graph()  # undirected
n = 0

for i in world_data['parsed_doc']:  # sure, it's inefficient, but it will do
    for j in world_data['parsed_doc']:
        if i != j:
            if not (raw_G.has_edge(j, i)):
                sim = i.similarity(j)
                raw_G.add_edge(i, j, weight=sim)
                n = n + 1

print("\n")
print(raw_G.number_of_nodes(), "nodes, and", raw_G.number_of_edges(), "edges created.")

#0.977
edges_to_kill = []
min_wt = 0.977  # this is our cutoff value for a minimum edge-weight

for n, nbrs in raw_G.adj.items():
    # print("\nProcessing origin-node:", n, "... ")
    for nbr, eattr in nbrs.items():
        # remove edges below a certain weight
        data = eattr['weight']
        if data < min_wt:
            # print('(%.3f)' % (data))
            # print('(%d, %d, %.3f)' % (n, nbr, data))
            # print("\nNode: ", n, "\n <-", data, "-> ", "\nNeighbour: ", nbr)
            edges_to_kill.append((n, nbr))

print("\n", len(edges_to_kill) / 2, "edges to kill (of", raw_G.number_of_edges(), "), before de-duplicating")

for u, v in edges_to_kill:
    if raw_G.has_edge(u, v):  # catches (e.g.) those edges where we've removed them using reverse ... (v, u)
        raw_G.remove_edge(u, v)

strong_G = raw_G
print(strong_G.number_of_edges())
print("\n")

# 20
# nx.draw(strong_G, node_size=18, edge_color='gray')

strong_G.remove_nodes_from(list(nx.isolates(strong_G)))

count = strong_G.number_of_nodes()

# 10 35 40 60
# default for this is 1/sqrt(n), but this will 'blow out' the layout for better visibility
equilibrium = 60 / sqrt(count)
pos = nx.fruchterman_reingold_layout(strong_G, k=equilibrium, iterations=2000)
nx.draw(strong_G, pos=pos, node_size=20, edge_color='gray')

plt.rcParams['figure.figsize'] = [16, 10]  # a better aspect ratio for labelled nodes

# 50
nx.draw(strong_G, pos, font_size=3, node_size=20, edge_color='gray', with_labels=False)
for p in pos:  # raise positions of the labels, relative to the nodes
    pos[p][1] -= 0.03
nx.draw_networkx_labels(strong_G, pos, font_size=8, font_color='k')

plt.show()

print(nx.info(strong_G))
