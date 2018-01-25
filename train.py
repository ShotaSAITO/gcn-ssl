import tensorflow as tf
import model
import utils
import numpy as np

A, D = utils.load_eu_core_matrices()
labels, k = utils.load_eu_core_true_label()

train_mask = utils.choose_mask(labels, k, 0.3)

#Classical graph regularization ssl
gssl = model.GraphSSL()
Y = gssl.load_Y(labels,train_mask)
F = gssl.graph_SSL(A, D, Y, 0.0001)
acc_ssl = gssl.acc_measure(F,labels)


#Graph CNN method
gcn = model.GCN()
num_points = len(labels)
adj = tf.placeholder(tf.float32, shape=(num_points, num_points))
deg = tf.placeholder(tf.float32, shape=(num_points, num_points))
num_clusters = tf.placeholder(tf.int32)   
embdng, pred, = gcn.gcn(adj, deg, k)

loss = gcn.get_loss(pred, labels, train_masks)

sess = tf.Session()

with tf.name_scope('summary'):
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)


optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
opt_op = optimizer.minimize(loss)


correct = 0
num_epochs = 1000
sess.run(tf.global_variables_initializer())
for i in range(num_epochs):
    _, a, _, loss_val = sess.run([opt_op, pred, embdng, loss], feed_dict={adj: A, deg: D, num_clusters: k})
    acc_gcn = gcn.acc_measure(a, lables)
    print('Correct GCN-SSL:', acc_gcn, 'Correct SSL:', acc_ssl)
    print('Loss:', loss_val)
