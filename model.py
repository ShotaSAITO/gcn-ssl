import tensorflow as tf
import numpy as np
import time


class GCN:

    def gcn_layer(self, H, A_tilde, num_nodes, output_dim, scope):
        
        with tf.variable_scope(scope):
            W = tf.Variable(tf.random_uniform((num_nodes, output_dim)))
            Z = tf.matmul(tf.matmul(A_tilde, H), W)
            H_out = tf.nn.relu(Z)
    
        return H_out


    def gcn(self, A, D, k):
    
        num_nodes = A.get_shape()[0].value
        num_classes = k  
        id = tf.eye(num_nodes)
        A_tilde = self.compute_filter(A,D)

        H_0 = id  # initial hidden state
    
        H_1 = self.gcn_layer(H_0, A_tilde, num_nodes, num_nodes, "L1")
        H_2 = self.gcn_layer(H_1, A_tilde, num_nodes, num_nodes, "L2")
        H_3 = self.gcn_layer(H_2, A_tilde, num_nodes, num_classes, "L3")
        y = H_3
    
        return H_2, y


    def compute_filter(self, A, D):
        num_nodes = A.get_shape()[0].value
        id = tf.eye(num_nodes)
        A = A + id
        Deg = D + id
        Deg = tf.sqrt(Deg)
        Deg = tf.reciprocal(Deg)
        Deg = tf.where(tf.is_inf(Deg), tf.zeros_like(Deg), Deg)
        A_tilde = tf.matmul(Deg, tf.matmul(A, Deg))
    
        return A_tilde


class GraphSSL:

    def graph_SSL(self, A,D,Y,alpha):
        num_nodes = A.shape[0]  # number of nodes in the graph, number of points
        D_sq = np.sqrt(D)
        D_sq_inv = np.diag(1/np.diagonal(D))
        D_sq_inv = np.where(np.isinf(D_sq_inv), np.zeros_like(D_sq_inv), D_sq_inv)
        A = A.astype('float64')
        L = np.matmul(D_sq_inv, np.matmul(A, D_sq_inv))
    
        id = np.eye(num_nodes)
    
        F = np.matmul(np.linalg.inv(id - alpha * L), Y)
    
        return F
    
    
    def load_Y(self, labels,train_masks):
        num_nodes = labels.shape[0]
        k = max(labels) + 1
        Y = np.zeros((num_nodes, k))
        Y[np.where(train_masks==1),labels[np.where(train_masks == 1)]] = 1
    
        return Y
    
    
    def acc_measure(self, F,labels):
        num_points = labels.shape[0]
        b = np.argmax(F, axis=1)
        correct = 0
        correct = np.sum(labels == b)/num_points
    
        return correct
