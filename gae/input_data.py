import pickle as pkl

import numpy as np


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):

    # load feature matrix
    with open('./data/{}_feats.pkl'.format(dataset), 'rb') as f1:
        features = pkl.load(f1)
    # load adjacency matrix
    # with open('./data/{}_adj.pkl'.format(dataset), 'rb') as f2:
    #     adj = pkl.load(f2, encoding='latin1')
    with open('./data/{}_new_adj.pkl'.format(dataset), 'rb') as f2:
        adj = pkl.load(f2, encoding='latin1')

    # load validation set and test set
    val_edges = np.load('./data/{}_val_edges.npy'.format(dataset))
    val_edges_false = np.load('./data/{}_val_edges_false.npy'.format(dataset))
    test_edges = np.load('./data/{}_test_edges.npy'.format(dataset))
    test_edges_false = np.load('./data/{}_test_edges_false.npy'.format(dataset))
    # labels = np.load('./data/{}_labels.npy'.format(dataset))
    with open('./data/{}_adj_train.pkl'.format(dataset), 'rb') as handle:
        adj_train = pkl.load(handle)

    # return new_adj, features
    return adj, features, adj_train, val_edges, val_edges_false, test_edges, test_edges_false  # , labels
