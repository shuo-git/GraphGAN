import numpy as np


def read_edges(train_filename, test_filename):
    graph = {}
    s_nodes = set()
    e_nodes = set()
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename)

    for edge in train_edges:
        s_nodes.add(edge[0])
        e_nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    for edge in test_edges:
        s_nodes.add(edge[0])
        e_nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []

    return max(max(s_nodes), max(e_nodes))+1, graph, s_nodes, e_nodes


def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = list(map(lambda x: list(map(int, x.split())), lines))
    return edges


def read_embeddings(filename, n_node, n_embed):
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
        embedding_matrix = np.random.rand(n_node, n_embed)
        for line in lines:
            emd = line.split()
            embedding_matrix[int(emd[0]), :] = list(map(float, emd[1:]))
        return embedding_matrix


def write_embeddings(filename, embedding_matrix, n_node, n_embed):
    index = np.array(range(n_node)).reshape(-1, 1)
    embedding_matrix = np.hstack([index, embedding_matrix])
    embedding_list = embedding_matrix.tolist()
    embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                     for emb in embedding_list]
    with open(filename, "w+") as f:
        lines = [str(n_node) + "\t" + str(n_embed) + "\n"] + embedding_str
        f.writelines(lines)
