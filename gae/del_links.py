import pickle as pkl

import numpy as np
import scipy.sparse


def edit_dataset(dataset):
    with open('./data/{}_feats.pkl'.format(dataset), 'rb') as f1:
        features = pkl.load(f1)
    # load adjacency matrix
    with open('./data/{}_adj.pkl'.format(dataset), 'rb') as f2:
        adj = pkl.load(f2, encoding='latin1')

    attributes = np.array(features.toarray())
    students = np.where(attributes[:, 0] == 1)
    # yale 0 - 4 correspond to the utility attribute student/faculty status (rochester 0 - 5)
    faculties = np.where((attributes[:, 3] == 1) | (attributes[:, 4] == 1))[0]

    adj_arr = adj.toarray()
    for i, c in enumerate(adj_arr):
        if i in faculties:
            # for x,y in enumerate(c):
            #    if (x in faculties) & (y==1):
            #     y=0
            c[faculties] = 0

    new_adj = scipy.sparse.csr_matrix(adj_arr)
    with open('./data/{}_new_adj.pkl'.format(dataset), 'wb') as f3:
        pkl.dump(new_adj, f3)

edit_dataset('yale')