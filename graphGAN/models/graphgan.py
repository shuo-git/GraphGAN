import os
import pickle
import numpy as np
import tensorflow as tf
import graphGAN.utils.common as utils
import tqdm


def generator(params, inputs):
    with tf.variable_scope("generator"):
        generator_embedding_matrix = tf.get_variable(name="embedding",
                                                     shape=params.init_emb_g.shape,
                                                     initializer=tf.constant_initializer(params.init_emb_g),
                                                     trainable=True)
        generator_bias_vector = tf.get_variable(name="bias",
                                                shape=[params.n_node],
                                                initializer=tf.constant_initializer(np.zeros([params.n_node])),
                                                trainable=True)
        node_id = inputs[0]
        node_neighbor_id = inputs[1]
        reward = inputs[2]

        node_embedding = tf.nn.embedding_lookup(generator_embedding_matrix, node_id)  # batch_size * n_embed
        node_neighbor_embedding = tf.nn.embedding_lookup(generator_embedding_matrix, node_neighbor_id)
        bias = tf.gather(generator_bias_vector, node_neighbor_id)
        score = tf.reduce_sum(node_embedding * node_neighbor_embedding, axis=1) + bias
        prob = tf.clip_by_value(tf.nn.sigmoid(score), 1e-5, 1)

        loss = -tf.reduce_mean(tf.log(prob) * reward) + params.lambda_gen * (
                tf.nn.l2_loss(node_neighbor_embedding) + tf.nn.l2_loss(node_embedding))
        all_score = tf.matmul(generator_embedding_matrix, generator_embedding_matrix, transpose_b=True) \
                    + generator_bias_vector
        return all_score, loss, generator_embedding_matrix


def discriminator(params, inputs):
    with tf.variable_scope("discriminator"):
        dis_embedding_matrix = tf.get_variable(name="embedding",
                                               shape=params.init_emb_d.shape,
                                               initializer=tf.constant_initializer(params.init_emb_d),
                                               trainable=True)
        dis_bias_vector = tf.get_variable(name="bias",
                                          shape=[params.n_node],
                                          initializer=tf.constant_initializer(np.zeros([params.n_node])),
                                          trainable=True)

        node_id = inputs[0]
        node_neighbor_id = inputs[1]
        label = inputs[2]

        node_embedding = tf.nn.embedding_lookup(dis_embedding_matrix, node_id)
        node_neighbor_embedding = tf.nn.embedding_lookup(dis_embedding_matrix, node_neighbor_id)
        bias = tf.gather(dis_bias_vector, node_neighbor_id)
        score = tf.reduce_sum(tf.multiply(node_embedding, node_neighbor_embedding), axis=1) + bias

        loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=score)) + params.lambda_dis * (
                tf.nn.l2_loss(node_neighbor_embedding) +
                tf.nn.l2_loss(node_embedding) +
                tf.nn.l2_loss(bias))
        score = tf.clip_by_value(score, clip_value_min=-10, clip_value_max=10)
        reward = tf.log(1 + tf.exp(score))
        return reward, loss, dis_embedding_matrix


class GraphGAN(object):
    def __init__(self, params):
        self.params = params

    @staticmethod
    def build_generator(params, inputs):
        all_score, loss, emb = generator(params, inputs)
        return all_score, loss, emb

    @staticmethod
    def build_discriminator(params, inputs):
        reward, loss, emb = discriminator(params, inputs)
        return reward, loss, emb

    @staticmethod
    def eval_recommend(scores, params):
        test_edges = utils.read_edges_from_file(params.test_edges)
        train_edges = utils.read_edges_from_file(params.train_edges)
        unwatched = {}
        watched = {}
        recommended = {}

        for e in train_edges:
            u = e[0]
            m = e[1]
            if u not in watched:
                watched[u] = set()
            watched[u].add(m)

        for e in test_edges:
            u = e[0]
            m = e[1]
            if u not in unwatched:
                unwatched[u] = set()
            unwatched[u].add(m)

        accuracy = []
        recall = []
        for u in tqdm.tqdm(unwatched.keys()):
            score_res = []
            cur_watched = watched[u]
            for m in range(1, params.n_movies):
                if m in cur_watched:
                    continue
                cur_score = scores[u][m]
                score_res.append((m, cur_score))
            score_res.sort(key=lambda x: x[1], reverse=True)
            recommended[u] = set(list(zip(*score_res[:params.top_k]))[0])
            cur_acc = len(unwatched[u] & recommended[u]) * 1.0 / params.top_k
            cur_rec = len(unwatched[u] & recommended[u]) * 1.0 / len(unwatched[u])
            accuracy.append(cur_acc)
            recall.append(cur_rec)

        accuracy_avg = np.mean(accuracy)
        recall_avg = np.mean(recall)

        return accuracy_avg, recall_avg




