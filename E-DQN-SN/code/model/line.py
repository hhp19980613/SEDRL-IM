import tensorflow as tf
import numpy as np
import argparse
from line_model import LINEModel
from line_utils import GraphData
import pickle
import time
import networkx as nx


def embedding_main(graph_file):
    parser = argparse.ArgumentParser()
    # print(parser)
    parser.add_argument('--embedding_dim',type=int, default=32)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_batches', default=3000)
    parser.add_argument('--total_graph', default=True)
    parser.add_argument('--graph_file', default=graph_file)

    # 统计有多少个节点
    edges = np.loadtxt(graph_file)
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)
    nodeNum = len(graph.nodes())

    # +1 是对应着虚拟节点
    parser.add_argument('--num_nodes', default= nodeNum)
    args = parser.parse_args()
    print('args:',args)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        # test(args)
        print("00000")


def train(args):
    data_loader = GraphData(graph_file=args.graph_file,num_nodes=args.num_nodes)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)
    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                if b% 200 == 0:
                    print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
            # if b % 1000 == 0 or b == (args.num_batches - 1):
            #     embedding = sess.run(model.embedding)
            #     normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            #     #print(normalized_embedding)
            #     pickle.dump(data_loader.embedding_mapping(normalized_embedding),
            #                 open('data/embedding_%s.pkl' % suffix, 'wb'))
            if b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)
                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                embedding_save_address = '../data/_embedding.txt'
                with open(embedding_save_address,'w') as f:
                    for i in range(args.num_nodes):
                        for j in range(args.embedding_dim):
                            f.write(str(normalized_embedding[i][j])+' ')
                        if i != (args.num_nodes-1):
                            f.write('\n')


# def test(args):
#     pass

if __name__ == '__main__':
    graph_file = '../data/test_graph.txt'
    # embedding_main(graph_file,32)
    embedding_main('../data/graphwithVN.txt')