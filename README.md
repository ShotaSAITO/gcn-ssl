# gcn-ssl
Implementation of GCN and regularization graph Semi-supervised learning.

## Implemetations

### Graph CNN based SSL
SSL based on neural network with multiple layers of graph CNNs. The implementation of [1].

### Graph Regularization based SSL
To compare with Graph CNN based SSL, the regularization method [2] is the classic. The implementation of [2].

## Data
You need to download the [Email-eu-core](https://snap.stanford.edu/data/email-Eu-core.html) dataset from [SNAP](https://snap.stanford.edu/index.html) in the same directory.

## References

[1] T. N. Kipf, M. Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks In Proc. ICLR

[2] D. Zhou, O. Bousquet, T. N. Lal, J. Watson, and B. Schölkopf. 2004. Learning with local and global consistency. In Proc. NIPS
