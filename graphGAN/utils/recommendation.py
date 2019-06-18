import sys
import numpy as np
import random
import pickle
import collections
import copy
import tqdm
import multiprocessing


def write_edges(edges, out_file):
    with open(out_file+'.txt', 'w') as fw:
        for (a, b) in edges:
            fw.write('{}\t{}\n'.format(a, b))
    with open(out_file+'.pkl', 'wb') as fw:
        pickle.dump(edges, fw)


def write_tree(tree, out_file):
    with open(out_file+'.pkl', 'wb') as fw:
        pickle.dump(tree, fw)


def load_pkl(filename):
    with open(filename+'.pkl', 'rb') as fr:
        return pickle.load(fr)


def prepare_dataset(raw_file):
    print("Loading raw input...")
    reserved_edges = []
    users = set()
    movies = set()

    with open(raw_file, 'r') as fr:
        for line in fr:
            infos = line.strip().split('::')
            user_id = int(infos[0])
            movie_id = int(infos[1])
            rating = int(infos[2])
            if int(rating) > 3:
                reserved_edges.append((user_id, movie_id))
                users.add(user_id)
                movies.add(movie_id)

    shift_v = max(movies)
    reserved_edges = list(map(lambda x: (shift_v+x[0], x[1]), reserved_edges))
    print("User id adds {}".format(shift_v))
    random.shuffle(reserved_edges)
    all_num = len(reserved_edges)
    test_num = int(all_num * 0.1)
    test_edges = reserved_edges[:test_num]
    train_edges = reserved_edges[test_num+1:]

    return train_edges, test_edges, shift_v


def extract_movie_graph(edges, neg_edges):
    user2movies = {}
    for (user, movie) in edges:
        if user in user2movies:
            user2movies[user].append(movie)
        else:
            user2movies[user] = [movie]

    for user in user2movies:
        user2movies[user] = list(np.unique(user2movies[user]))

    for (user, movie) in neg_edges:
        if user in user2movies and movie in user2movies[user]:
            user2movies[user].remove(movie)

    movie_graph = {}
    num_movie_edge = 0
    for user in user2movies:
        movies = user2movies[user]
        for m1 in movies:
            if not (m1 in movie_graph):
                movie_graph[m1] = []
            for m2 in movies:
                if m1 == m2:
                    continue
                if m2 not in movie_graph[m1]:
                    movie_graph[m1].append(m2)
                    num_movie_edge += 1

    print("Add {} shortcut edges".format(num_movie_edge))
    return movie_graph, user2movies


def BFS(graph, root):
    tree = {}
    tree[root] = [root]
    used_nodes = set()
    queue = collections.deque([root])
    while len(queue) > 0:
        cur_node = queue.popleft()
        used_nodes.add(cur_node)
        for sub_node in graph[cur_node]:
            if sub_node not in used_nodes:
                tree[cur_node].append(sub_node)
                tree[sub_node] = [cur_node]
                queue.append(sub_node)
                used_nodes.add(sub_node)
    return tree


def meta_func(kwargs):
    ns = kwargs[0]
    movie_graph = kwargs[1]
    user2movies = kwargs[2]
    u2t = {}
    for user in tqdm.tqdm(ns):
        temp_graph = copy.deepcopy(movie_graph)
        temp_graph[user] = user2movies[user]
        for m in user2movies[user]:
            temp_graph[m].append(user)
        u2t[user] = BFS(temp_graph, user)
    return u2t


def construct_tree(train_edges, test_edges):
    print("Constructing movie graph...")
    movie_graph, user2movies = extract_movie_graph(train_edges, test_edges)
    write_tree(movie_graph, 'movie_graph')
    write_tree(user2movies, 'u2m')
    movie_graph = load_pkl('movie_graph')
    user2movies = load_pkl('u2m')
    print("Constructing BFS trees...")

    n_node = len(user2movies)
    cores = multiprocessing.cpu_count() // 2
    print("{} cores are working at the same time...".format(cores))
    pool = multiprocessing.Pool(cores)
    nodes = list(user2movies.keys())
    new_nodes = []
    n_node_per_core = n_node // cores
    for i in range(cores):
        if i != cores - 1:
            new_nodes.append((nodes[i * n_node_per_core: (i + 1) * n_node_per_core],
                              movie_graph, user2movies))
        else:
            new_nodes.append((nodes[i * n_node_per_core:],
                              movie_graph, user2movies))
    user2trees = {}
    trees_result = pool.map(meta_func, new_nodes)
    for tree in trees_result:
        user2trees.update(tree)
    return user2trees


if __name__ == "__main__":
    train_edges, test_edges, shift_v = prepare_dataset(sys.argv[1])
    write_edges(train_edges, 'train_edges')
    write_edges(test_edges, 'test_edges')
    user2trees = construct_tree(train_edges, test_edges)
    write_tree(user2trees, 'train_trees')

