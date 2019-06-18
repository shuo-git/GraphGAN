import sys
import numpy as np
import random
import pickle
import collections
import copy
import tqdm
import multiprocessing
import graphGAN.utils.common as utils
import graphGAN.utils.recommendation as recommendation


def meta_func(kwargs):
    ns = kwargs[0]
    cur_graph = kwargs[1]
    trees = {}
    for node in tqdm.tqdm(ns):
        trees[node] = recommendation.BFS(cur_graph, node)
    return trees


if __name__ == '__main__':
    train_edges = utils.read_edges_from_file(sys.argv[1])
    test_edges = utils.read_edges_from_file(sys.argv[2])

    graph = {}
    nodes = set()

    for edge in train_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    for edge in test_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []

    cores = multiprocessing.cpu_count() // 2
    print("{} cores are working at the same time...".format(cores))
    pool = multiprocessing.Pool(cores)
    nodes = list(nodes)
    n_node = len(nodes)
    new_nodes = []
    n_node_per_core = n_node // cores

    for i in range(cores):
        if i != cores - 1:
            new_nodes.append((nodes[i * n_node_per_core: (i + 1) * n_node_per_core], graph))
        else:
            new_nodes.append((nodes[i * n_node_per_core:], graph))

    all_trees = {}
    trees_result = pool.map(meta_func, new_nodes)
    for tree in trees_result:
        all_trees.update(tree)

    recommendation.write_tree(all_trees, "train_trees")
