import argparse
import os

import tensorflow as tf
import numpy as np
import graphGAN.utils.common as utils
import graphGAN.utils.data as data
import pickle
import graphGAN.models.graphgan as graphgan
import tqdm


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Train a GraphGAN',
        usage='trainer.py [<args>] [-h | --help]'
    )

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--parameters", type=str, default="")

    return parser.parse_args(args)


def default_parameters():
    params = tf.contrib.training.HParams(
        data_dir="",
        task="recommendation",
        emb_generator="gen.emb",
        emb_discriminator="dis.emb",
        record="record",
        log_dir="",
        train_edges="train_edges.txt",
        test_edges="test_edges.txt",
        test_neg_edges="test_neg_edges.txt",
        train_trees="train_trees.pkl",
        n_node=1000,
        graph={},
        s_nodes=set(),
        e_nodes=set(),
        trees={},
        batch_size_gen=64,  # batch size for the generator
        batch_size_dis=64,  # batch size for the discriminator
        lambda_gen=1e-5,  # l2 loss regulation weight for the generator
        lambda_dis=1e-5,  # l2 loss regulation weight for the discriminator
        n_sample_gen=20,  # number of samples for the generator
        lr_gen=1e-3,  # learning rate for the generator
        lr_dis=1e-3,  # learning rate for the discriminator
        window_size=2,
        n_epochs=20,  # number of outer loops
        n_epochs_gen=30,  # number of inner loops for the generator
        n_epochs_dis=30,  # number of inner loops for the discriminator
        gen_interval=30,  # sample new nodes for the generator for every gen_interval iterations
        dis_interval=30,  # sample new nodes for the discriminator for every dis_interval iterations
        update_ratio=1,  # updating ratio when choose the trees
        save_steps=1,
        n_movies=3953,
        n_emb=50,
        top_k=10,
        max_to_save=100,
        pretrain_emb_filename_d="pre_train.emb",
        pretrain_emb_filename_g="pre_train.emb",
        init_emb_d=np.array([0]),
        init_emb_g=np.array([0])
    )

    return params


def import_params(log_dir, params):
    p_name = os.path.join(os.path.abspath(log_dir), 'params.jason')

    if not tf.gfile.Exists(p_name):
        return params

    with tf.gfile.Open(p_name) as fr:
        tf.logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fr.readline()
        params.parse_json(json_str)

    return params


def override_params(params, args):
    params.data_dir = args.data_dir
    params.log_dir = args.log_dir
    params.parse(args.parameters)

    params.emb_generator = params.log_dir + '/' + params.emb_generator
    params.emb_discriminator = params.log_dir + '/' + params.emb_discriminator
    params.record = params.log_dir + '/' + params.record
    params.train_edges = params.data_dir + '/' + params.train_edges
    params.test_edges = params.data_dir + '/' + params.test_edges
    params.test_neg_edges = params.data_dir + '/' + params.test_neg_edges
    params.train_trees = params.data_dir + '/' + params.train_trees
    params.pretrain_emb_filename_d = params.data_dir + '/' + params.pretrain_emb_filename_d
    params.pretrain_emb_filename_g = params.data_dir + '/' + params.pretrain_emb_filename_g

    tf.logging.info("Reading edges...")
    params.n_node, params.graph, params.s_nodes, params.e_nodes = \
        utils.read_edges(params.train_edges, params.test_edges)

    if params.task != "recommendation":
        params.s_nodes = params.s_nodes | params.e_nodes

    with open(params.train_trees, 'rb') as fr:
        tf.logging.info("Loading BFS trees...")
        params.trees = pickle.load(fr)

    if os.path.isfile(params.pretrain_emb_filename_d):
        tf.logging.info("Loading pre-trained discriminator embedding...")
        params.init_emb_d = utils.read_embeddings(params.pretrain_emb_filename_d, params.n_node, params.n_emb)
    else:
        params.init_emb_d = np.random.randn(params.n_node, params.n_emb) / float(params.n_emb)

    if os.path.isfile(params.pretrain_emb_filename_g):
        tf.logging.info("Loading pre-trained generator embedding...")
        params.init_emb_g = utils.read_embeddings(params.pretrain_emb_filename_g, params.n_node, params.n_emb)
    else:
        params.init_emb_g = np.random.randn(params.n_node, params.n_emb) / float(params.n_emb)

    return params


def print_variables():
    all_weights = {v.name: v for v in tf.trainable_variables()}
    total_size = 0

    for v_name in sorted(list(all_weights)):
        v = all_weights[v_name]
        tf.logging.info("%s\tshape    %s", v.name.ljust(80),
                        str(v.shape).ljust(20))
        v_size = np.prod(np.array(v.shape.as_list())).tolist()
        total_size += v_size
    tf.logging.info("Total trainable variables size: %d", total_size)


def after_run(params, sess, model, g_score, d_score, best_acc, epoch, g_emb, d_emb):
    gen_all_score_v = sess.run(g_score)
    dis_all_score_v = sess.run(d_score)
    if params.task == "recommendation":
        gen_accuracy, gen_2 = model.eval_recommend(gen_all_score_v, params)
        dis_accuracy, dis_2 = model.eval_recommend(dis_all_score_v, params)
    elif params.task == "classification":
        utils.write_embeddings(params.emb_generator, sess.run(g_emb), params.n_node, params.n_emb)
        utils.write_embeddings(params.emb_discriminator, sess.run(d_emb), params.n_node, params.n_emb)
        return 0.0
    else:
        gen_accuracy, gen_2 = model.eval_link_prediction(gen_all_score_v, params)
        dis_accuracy, dis_2 = model.eval_link_prediction(dis_all_score_v, params)

    if dis_accuracy > best_acc:
        best_acc = dis_accuracy
        utils.write_embeddings(params.emb_generator, sess.run(g_emb), params.n_node, params.n_emb)
        utils.write_embeddings(params.emb_discriminator, sess.run(d_emb), params.n_node, params.n_emb)

    with open(params.record, 'a+') as fw:
        fw.write('gen\t{}\t{:.10f}\t{:.10f}\n'.format(epoch, gen_accuracy, gen_2))
        fw.write('dis\t{}\t{:.10f}\t{:.10f}\n'.format(epoch, dis_accuracy, dis_2))

    return best_acc


def train(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    params = default_parameters()
    params = import_params(args.log_dir, params)
    params = override_params(params, args)

    model = graphgan.GraphGAN(params)

    node_id = tf.placeholder(tf.int32, shape=[None])
    node_neighbor_id = tf.placeholder(tf.int32, shape=[None])
    reward = tf.placeholder(tf.float32, shape=[None])
    label = tf.placeholder(tf.float32, shape=[None])

    tf.logging.info("Building generator...")
    gen_all_score, gen_loss, gen_emb = model.build_generator(params, [node_id, node_neighbor_id, reward])
    g_optimizer = tf.train.AdamOptimizer(params.lr_gen)
    g_updates = g_optimizer.minimize(gen_loss)
    tf.logging.info("Building discriminator...")
    dis_reward, dis_loss, dis_emb = model.build_discriminator(params, [node_id, node_neighbor_id, label])
    d_optimizer = tf.train.AdamOptimizer(params.lr_dis)
    d_updates = d_optimizer.minimize(dis_loss)

    print_variables()

    saver = tf.train.Saver(max_to_keep=params.max_to_save)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session(config=sess_config)
    sess.run(init_op)

    checkpoint = tf.train.get_checkpoint_state(params.log_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        tf.logging.info("loading the checkpoint: %s" % checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

    utils.write_embeddings(params.emb_generator, sess.run(gen_emb), params.n_node, params.n_emb)
    utils.write_embeddings(params.emb_discriminator, sess.run(dis_emb), params.n_node, params.n_emb)

    gene_all_score = tf.matmul(gen_emb, gen_emb, transpose_b=True)
    disc_all_score = tf.matmul(dis_emb, dis_emb, transpose_b=True)

    best_acc = 0.0

    best_acc = after_run(params, sess, model, gene_all_score, disc_all_score,
                         best_acc, 0, gen_emb, dis_emb)

    for epoch in range(1, params.n_epochs+1):
        tf.logging.info("epoch %d" % epoch)
        if epoch % params.save_steps == 0:
            saver.save(sess, params.log_dir + "/model-{}".format(epoch))

            # D-steps
            center_nodes = []
            neighbor_nodes = []
            labels = []
            all_score_v = sess.run(gen_all_score)

            for d_epoch in tqdm.tqdm(range(params.n_epochs_dis)):
                # generate new nodes for the discriminator for every dis_interval iterations
                if d_epoch % params.dis_interval == 0:
                    center_nodes, neighbor_nodes, labels = data.prepare_data_for_d(params, all_score_v)
                # training
                train_size = len(center_nodes)
                start_list = list(range(0, train_size, params.batch_size_dis))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + params.batch_size_dis
                    sess.run(d_updates, feed_dict={node_id: np.array(center_nodes[start:end]),
                                                   node_neighbor_id: np.array(neighbor_nodes[start:end]),
                                                   label: np.array(labels[start:end])})

            # G-steps
            node_1 = []
            node_2 = []
            reward_v = []
            for g_epoch in tqdm.tqdm(range(params.n_epochs_gen)):
                if g_epoch % params.gen_interval == 0:
                    all_score_v = sess.run(gen_all_score)
                    node_1, node_2 = data.prepare_data_for_g(params, all_score_v)
                    reward_v = sess.run(dis_reward, feed_dict={node_id: np.array(node_1),
                                                               node_neighbor_id: np.array(node_2)})

                # training
                train_size = len(node_1)
                start_list = list(range(0, train_size, params.batch_size_gen))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + params.batch_size_gen
                    sess.run(g_updates, feed_dict={node_id: np.array(node_1[start:end]),
                                                   node_neighbor_id: np.array(node_2[start:end]),
                                                   reward: np.array(reward_v[start:end])})
            # Evaluation
            best_acc = after_run(params, sess, model, gene_all_score, disc_all_score,
                                 best_acc, epoch, gen_emb, dis_emb)


if __name__ == "__main__":
    train(parse_args())
