import numpy as np
import pickle
import scipy.sparse as sp

def normalize_features(X):
    X_np = X.toarray()
    row_sum = np.sum(X_np, axis = 1)
    row_inv = row_sum**-1
    inf_indices = np.where(row_inv == np.inf)[0]
    row_inv[inf_indices] = 0.0
    r_mat_inv = sp.diags(row_inv)
    res = r_mat_inv.dot(X)
    return res

def normalize_adj(adj):
    adj = sp.csr_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def save_data(file_path, save):
    with open(file_path, "wb") as f:
        data = pickle.dump(save, f)
    return data