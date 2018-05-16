import tensorflow as tf
from tensorflow.python.ops import variable_scope

class SimpleNN(object):

    def __init__(self, w_shape, batch_size, reuse=None):

        with variable_scope.variable_scope("FullyNN", reuse=reuse):
            self.num_layers = len(w_shape)
            self.ws = []
            self.bs = []
            self.ts = []
            self.tvars = []
            self.grads = []
            self.input = tf.placeholder(dtype=tf.float32, shape=[batch_size, 5])
            #self.input = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784])
            self.lr = tf.placeholder(dtype=tf.float32, shape=())
            #self.target = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10])
            self.target = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1])
            def leaky_relu(tens):
                return tf.where(tens < tf.zeros_like(tens, dtype=tf.float32), 0.05 * tens, tens)

            t_prev = self.input
            for l in range(self.num_layers):
                w_scale = 1 / float(w_shape[l][0])

                w = tf.get_variable("w_"+str(l), dtype=tf.float32, shape=w_shape[l], initializer=tf.random_normal_initializer(stddev=w_scale))
                b = tf.get_variable("b_"+str(l), dtype=tf.float32, shape=w_shape[l][1], initializer=tf.random_normal_initializer(-w_scale, w_scale))
                u = tf.matmul(t_prev, w) + b
                self.ws.append(w)
                self.bs.append(b)

                if l == self.num_layers - 1:
                    t = tf.nn.sigmoid(u)
                else:
                    t = leaky_relu(u)

                self.ts.append(t)
                t_prev = t

            logits = self.ts[-1]
            ce = - self.target * tf.log(logits)
            self.cost = loss = tf.reduce_mean(tf.reduce_sum(ce, axis=[1]), axis=[0])
            
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.target, 1))
            self.accuaracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            self.tvars = [a for a in self.ws]
            self.tvars.extend(self.bs)
            self.grads_w = tf.gradients(loss, self.ws)
            self.grads_b = tf.gradients(loss, self.bs)
            self.grads = [a for a in self.grads_w]
            self.grads.extend(self.grads_b)

            # optimizer = tf.train.AdamOptimizer()
            # self.eval_op = optimizer.apply_gradients(zip(self.grads, self.tvars))

            optimizer_gd = tf.train.GradientDescentOptimizer(self.lr)
            optimizer_ad = tf.train.AdamOptimizer()
            self.op_gd = optimizer_gd.apply_gradients(zip(self.grads, self.tvars))
            self.op_ad = optimizer_ad.apply_gradients(zip(self.grads, self.tvars))

            print("model constructed")


    def assign_vars(self, tensor_values, sess):
        assign_ops = []
        for layer_k, w_value, b_value in tensor_values:
            assign_op_w_k = tf.assign(self.ws[layer_k], w_value)
            assign_op_b_k = tf.assign(self.bs[layer_k], b_value)
            assign_ops.append(assign_op_w_k)
            assign_ops.append(assign_op_b_k)
        sess.run(assign_ops)

    def sess_run_op(self, sess, eval_tensor, feed_dict, lr):

        op_eval = [x for x in eval_tensor]
        if lr < 0.0:
            op_eval.append(self.op_ad)
        else:
            op_eval.append(self.op_gd)
        return sess.run(op_eval, feed_dict=feed_dict)

