import networkx as nx
import numpy as np
import tensorflow as tf
import random

def load_eu_core_matrices():
    G = nx.read_edgelist('email-Eu-core.txt')

    A = nx.adjacency_matrix(G)
    L = nx.laplacian_matrix(G)
    D = L + A

    A = A.toarray()
    D = D.toarray()

    return A, D


def load_eu_core_true_label():
    """
    Output
    label: vector
    num_of_cluster: number of clusters
    """

    labels = np.loadtxt("email-Eu-core-department-labels.txt")
    labels = labels[:,1]
    labels = labels.astype(int)

    num_clusters = int(np.max(labels) + 1)

    return labels, num_clusters


def load_karate_matrices():
    G = nx.karate_club_graph()

    A = nx.adjacency_matrix(G)
    L = nx.laplacian_matrix(G)
    D = L + A

    A = A.toarray()
    D = D.toarray()

    return A, D


def load_karate_true_label():
    
    labels = [0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 3, 3, 3, 2, 3, 3, 2, 2, 3, 2, 2]
    labels = np.array(labels)
    labels = labels.astype('int32')
    num_clusters = int(max(labels) + 1)
    return labels, num_clusters


def choose_mask(labels,k,proportion):
    num_nodes = labels.shape[0]
    
    num_uniq_known_labels = 0
    trial = 0
    num_known_labels = int(num_nodes * proportion)


    ##As some of lables assigned to very small amount of nodes, we require a certain part of the labels assigined to nodes as a training labels
    while num_uniq_known_labels < k * 3/4 and trial < 100:
        train_mask = np.zeros_like(labels)
        task_control = np.zeros_like(labels)  
        mask = np.random.randint(0,num_nodes,num_known_labels).tolist()
        task_control[mask] = labels[mask]
        
        train_mask[mask] = 1

        num_uniq_known_labels = np.unique(task_control).size
        trial += 1

    if trial == 100:
        print('I guess we need more proportion')
        raise 

    return train_mask
