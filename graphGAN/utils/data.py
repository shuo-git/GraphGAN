import os
import collections
import numpy as np
import tensorflow as tf


def prepare_data_for_d(params, all_score):
    center_nodes = []
    neighbor_nodes = []
    labels = []
    for i in params.s_nodes:
        if np.random.rand() < params.update_ratio:
            pos = params.graph[i]
            neg, _ = sample(i, params.trees[i], len(pos), all_score, for_d=True)
            if len(pos) != 0 and neg is not None:
                # positive samples
                center_nodes.extend([i] * len(pos))
                neighbor_nodes.extend(pos)
                labels.extend([1] * len(pos))

                # negative samples
                center_nodes.extend([i] * len(pos))
                neighbor_nodes.extend(neg)
                labels.extend([0] * len(neg))
    return center_nodes, neighbor_nodes, labels


def prepare_data_for_g(params, all_score):
    paths = []
    for i in params.s_nodes:
        if np.random.rand() < params.update_ratio:
            samples, paths_from_i = sample(i, params.trees[i], params.n_sample_gen, all_score, for_d=False)
            if paths_from_i is not None:
                paths.extend(paths_from_i)
    node_pairs = list(map(lambda x: get_node_pairs_from_path(x, params.window_size), paths))
    node_1 = []
    node_2 = []
    for i in range(len(node_pairs)):
        for pair in node_pairs[i]:
            node_1.append(pair[0])
            node_2.append(pair[1])

    return node_1, node_2


def sample(root, tree, sample_num, all_score, for_d):
    samples = []
    paths = []
    n = 0

    while len(samples) < sample_num:
        current_node = root
        previous_node = -1
        paths.append([])
        is_root = True
        paths[n].append(current_node)
        while True:
            node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
            is_root = False
            if len(node_neighbor) == 0:  # the tree only has a root
                return None, None
            if for_d:  # skip 1-hop nodes (positive samples)
                if node_neighbor == [root]:
                    # in current version, None is returned for simplicity
                    return None, None
                if root in node_neighbor:
                    node_neighbor.remove(root)
            relevance_probability = all_score[current_node, node_neighbor]
            relevance_probability = softmax(relevance_probability)
            next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
            paths[n].append(next_node)
            if next_node == previous_node:  # terminating condition
                samples.append(current_node)
                break
            previous_node = current_node
            current_node = next_node
        n = n + 1
    return samples, paths


def get_node_pairs_from_path(path, window_size):
    path = path[:-1]
    pairs = []
    for i in range(len(path)):
        center_node = path[i]
        for j in range(max(i - window_size, 0), min(i + window_size + 1, len(path))):
            if i == j:
                continue
            node = path[j]
            pairs.append([center_node, node])
    return pairs


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()
