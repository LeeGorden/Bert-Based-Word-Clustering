# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 21:46:00 2016
Draw Network and calculate related indicators of nodes in network

@author: Li Gorden
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


# ------------------------Calculation of Similarity and Definition of Interaction--------------------------
def find_interaction(x, method='cosine', threshold=0.2, feature=True):
    """
    Function:
    find the interaction between features
    Variable:
    x-DataFrame of raw data with row = samples and col = features(excluding y-label)
    method-{'circle', 'random', 'shell', 'spring', 'spectral'}
           the method to calculate distance between samples/features in input x
    threshold-the threshold of distance to be defined as 'has interaction'
    feature-if = True, meaning to calculating distance among features, else among samples
    Return:
    list of sample index/feature names of x which have interaction
    """
    # Initialize variable to be used
    interact_list = list()
    # When feature=True, find interaction between features, else find interaction between samples
    if feature:
        nodes_name = x.columns
        x = x.T
    else:
        nodes_name = x.index
    # Calculate the distance based on method input
    if method == "cosine":
        print('Calculating distance using cosine distance')
        dist = pdist(x, metric='cosine')
    elif method == "euclidean":
        print('Calculating distance using euclidean distance')
        dist = pdist(x, metric='euclidean')
    elif method == "mahalanobis":
        print('Calculating distance using mahalanobis distance')
        dist = pdist(x, metric='mahalanobis')
    dist_max = np.max(dist)
    dist_min = np.min(dist)
    # Draw distribution chart of cosine distance
    plt.figure()
    plt.hist(dist)
    plt.title(method + ' distance')
    plt.xlabel(method + ' distance')
    plt.ylabel('count')
    plt.annotate('Mean: ' + str(round(np.mean(dist), 3)), xy=(0.75 * dist_max, len(dist) / 5 * 1),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 1))
    plt.annotate('Max: ' + str(round(dist_max, 3)), xy=(0.75 * dist_max, len(dist) / 5 * 0.9),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 0.9))
    plt.annotate('75%: ' + str(round(np.percentile(dist, 75), 3)), xy=(0.75 * dist_max, len(dist) / 5 * 0.8),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 0.8))
    plt.annotate('50%: ' + str(round(np.percentile(dist, 50), 3)), xy=(0.75 * dist_max, len(dist) / 5 * 0.7),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 0.7))
    plt.annotate('25%: ' + str(round(np.percentile(dist, 25), 3)), xy=(0.75 * dist_max, len(dist) / 5 * 0.6),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 0.6))
    plt.annotate('Min: ' + str(round(dist_min, 3)), xy=(0.75 * dist_max, len(dist) / 5 * 0.5),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 0.5))
    plt.show()
    # Select object satisfying the threshold, define them as 'has interaction' and record in a list
    dist = squareform(dist)
    for i in range(0, len(dist)):
        for j in range(i + 1, len(dist)):
            # ??????????????????????????????, ?????????feature_name??????interact_list???
            if dist[i][j] <= threshold:
                # Since we are going to create edges for a DiGraph using interact_list
                # Thus put both [i, j] and [j, i] into interact_list
                interact_list.append([nodes_name[i], nodes_name[j]])
    return interact_list


# -----------------------------Draw Topology Graph-------------------------------------------
def draw_topology_graph(x, edges, layout='random', feature=True, label=True):
    """
    Function:
    Return the topology graph and corresponding centrality
    Variable:
    x-DataFrame, data input
    edges-edges of the graph defined as 'has interaction'
    layout-g.pos, type of topology used
    feature-if = True, meaning to calculating distance among features, else among samples
    Return:
    graph and centrality features
    """
    # Initialize variable to be used
    if feature:
        nodes_name = x.columns
        x = x.T
    else:
        nodes_name = x.index
    # ???????????????, ????????????sequence data,
    # ????????????????????????feature???????????????feature????????????feature??????feature???interaction?????????
    g = nx.Graph()  # ???????????????g = nx.DiGraph()
    # Way to add node one-by-one: g.add_node(0)
    # Add_nodes_from([])?????????list??????????????????
    g.add_nodes_from(nodes_name)
    # ???node??????data, ?????????feature????????????samples????????????node???, ??????g.nodes[i]???????????????????????????
    for node_name in g.nodes():
        g.nodes[node_name].update(x.loc[node_name])
    # Graph????????????????????????interaction????????????(????????????????????????????????????????????????????????????)
    g.add_edges_from(edges)
    # Set layout for network graph
    if layout == 'random':
        g.pos = nx.random_layout(g)  # default to scale=1
    elif layout == 'spring':
        g.pos = nx.spring_layout(g)
    elif layout == 'shell':
        g.pos = nx.shell_layout(g)
    elif layout == 'circle':
        g.pos = nx.circular_layout(g)
    elif layout == 'spectral':
        g.pos = nx.spectral_layout(g)
    elif layout == 'kamada_kawai':
        g.pos = nx.kamada_kawai_layout(g)
    # Plot the network graph
    plt.figure()
    nx.draw(g, pos=g.pos,
            node_size=100, node_color='y', with_labels=label, font_size=12,
            alpha=0.5, width=1, edge_color='b', style='solid')
    plt.title('Network_' + layout)
    plt.show()
    # Output centrality information
    # Degree_centrality???????????????????????????????????????, =??????????????????edge/???edge???
    degree_c = pd.DataFrame({'Degree Centrality': nx.degree_centrality(g)})
    # Closeness_centrality????????????????????????????????????, =(n-1) * 1/???????????????edge????????????????????????????????????(?????????????????????????????????1)
    closeness_c = pd.DataFrame({'Closeness Centrality': nx.closeness_centrality(g)})
    # Betweenness_centrality?????????????????????????????? ,=??????????????????????????????????????????
    betweenness_c = pd.DataFrame({'Betweeness Centrality': nx.betweenness_centrality(g)})
    load_c = pd.DataFrame({'Load Centrality': nx.load_centrality(g)})
    # Harmonic_centrality-???????????????
    harmonic_c = pd.DataFrame({'Harmonic Centrality': nx.harmonic_centrality(g)})
    # Combining Network Information with Raw Feature Data
    centrality_data = degree_c.join(closeness_c)
    centrality_data = centrality_data.join(betweenness_c)
    centrality_data = centrality_data.join(load_c)
    centrality_data = centrality_data.join(harmonic_c)
    return centrality_data, g, g.pos
