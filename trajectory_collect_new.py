# -*- coding:utf-8 -*-
__author__ = 'st491'
from model.network import SimpleNN
import tensorflow as tf
#from data.DataHelper import trn, get_full_data2, get_subsample_data
from data.DataHelper import get_train_data, get_train_data
import numpy as np
import numpy.random as nr
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import scipy.io as sio   
import pickle
session = tf.Session()
default_g = tf.Graph().as_default()
session.as_default()

# 网络结构声明
#model = SimpleNN([[784, 128], [128, 128], [128, 128], [128, 128], [128, 10]], 60000)
model = SimpleNN([[5,3], [3, 3], [3, 3], [3, 1]], 4000)
model_sgd = SimpleNN([[5,3], [3, 3], [3, 3], [3, 1]], 1,reuse=True)


# 噪音信号比等级声明
noise_levels = [0.05, 0.2, 0.5, 1.0, 2.0]

# 采样(w, x)计算带噪音的t的采样数（最好等同于Mnist数据大小=60000）
noisy_mi_sampling_num = 200

# 单轮噪音引入所在的bgd迭代轮数
bgd_stairs = [100, 300, 500]

# lr小于0时，bgd使用adam优化器，大于0时使用gradient descent优化器。tuple=(小于该轮数时, 使用的lr值)
lr_adjust = [(100, -0.1), (140, 0.5)]

# 是否真正计算MI
enable_mi_calculation = True

tf.global_variables_initializer().run(session=session)

def zip2(*args):
    # zip2(x, y) takes the lists of vectors and makes it a list of vectors in a joint space
    # E.g. zip2([[1], [2], [3]], [[4], [5], [6]]) = [[1, 4], [2, 5], [3, 6]]
    return [sum(sublist, []) for sublist in zip(*args)]

def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    N = len(points)
    tree = ss.cKDTree(points)
    avg = 0.
    for i in range(N):
        dist = dvec[i]
        # subtlety, we don't include the boundary point,
        # but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(points[i], dist - 1e-15, p=float('inf')))
        avg += digamma(num_points) / N
    return avg

#def load_data(name):
    #d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), name + '.mat'))
    #F = d['F']
    #y = d['y']
    #C = type('type_C', (object,), {})
    #data_sets = C()
    #data_sets.data = F
    #data_sets.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)



def mi(x, y, k=3, base=2):
    """ Mutual information of x and y
        x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    intens = 0  # small noise to break degeneracy, see doc.()
    #x = [list(x)]
    #y = [list(y)]
    x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
    y = [list(p + intens * nr.rand(len(y[0]))) for p in y]
    points = zip2(x, y)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = ss.cKDTree(points)
    dvec = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in points]
    a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
    return (-a - b + c + d) / log(base)


def get_MI(t_val, x_val, y_val):
    if enable_mi_calculation:
        itx = mi(x_val, t_val)
        ity = mi(y_val, t_val)
        #itx = bin_calc_information2(x_val,t_val,0.05)
        return itx,ity
    else:
        return 0.0, 0.0

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse


def bin_calc_information2(labelixs, layerdata, binsize):
    # This is even further simplified, where we use np.floor instead of digitize
    def get_h(d):
        digitized = np.floor(d / binsize).astype('int')
        p_ts, _ = get_unique_probs( digitized )
        return -np.sum(p_ts * np.log(p_ts))

    H_LAYER = get_h(layerdata)
    #H_LAYER_GIVEN_OUTPUT = 0
    #for label, ixs in labelixs.items():
        #H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_h(layerdata[ixs,:])
    #return H_LAYER, H_LAYER - H_LAYER_GIVEN_OUTPUT
    return H_LAYER


def get_layers_MI(t_tensor, x_val, y_val, iter_num):

    xts = []
    yts = []
    for layer_k in range(len(t_tensor)):
        xt, yt = get_MI(t_tensor[layer_k], x_val, y_val)
        xts.append(xt)
        yts.append(yt)
    iter_num_list = [iter_num for _ in range(len(xts))]
    return xts, yts, iter_num_list


def get_lr(iter_num):
    for iter_threashold, lr_value in lr_adjust: 
        if iter_num < iter_threashold:
            return lr_value   


def collect_trajectory(breaking_points, noise_levels):

    def bgd_trajectory(start_iter_num, run_iter):

        mi_tx = []
        mi_ty = []
        iters = []
        #full_inp, full_tgt = get_full_data2()
        full_inp = trn.X
        full_tgt = trn.Y_oh
        #full_inp, full_tgt = get_train_data()
        feed_dict = {model.input:full_inp, model.target:full_tgt}
        
        
        for i in range(run_iter):
            lr = get_lr(i+start_iter_num)
            eval_list = [x for x in model.ts]
            eval_list.append(model.cost)
            eval_list.append(model.accuaracy)
            ts_op = model.sess_run_op(session, eval_list, feed_dict, lr)
            ts_tensor = ts_op[:-3]
            cost = ts_op[-3] 
            accuaracy = ts_op[-2]
            iters.append(start_iter_num + i)
            
            if i % 10 == 0:
                print (i,cost,accuaracy)    
                xts, yts, _ = get_layers_MI(ts_tensor, full_inp, full_tgt, start_iter_num+i)
                mi_tx.append(xts)
                mi_ty.append(yts)
        return mi_tx, mi_ty, iters

    def zero_noise_trajectory_point(start_vars, start_var_grads, start_iter_num, lr):
        start_ws, start_bs = start_vars
        start_grads_w, start_grads_b = start_var_grads

        #feed_dict = {model.input:full_input, model.target:full_target, model.lr:lr}
        # zero-noise MI
        assign_vars = []
        for layer_k in range(len(start_ws)):
            zero_noise_w_layer_k = start_ws[layer_k] - lr * start_grads_w[layer_k]
            zero_noise_b_layer_k = start_bs[layer_k] - lr * start_grads_b[layer_k]
            assign_vars.append((layer_k, zero_noise_w_layer_k, zero_noise_b_layer_k))
        model.assign_vars(assign_vars, session)

        ts_layers = session.run(model.ts, feed_dict=feed_dict)

        print ("calculating zero_noise mi")
        xts, yts, noise_iters = get_layers_MI(ts_layers, trn.X, trn.Y_oh, start_iter_num)

        return xts, yts, start_iter_num

    def noised_trajectory_point(start_tvars, start_grads, noise_std_klayers_llevels, start_iter_num, lr):
        start_ws, start_bs = start_tvars
        start_grads_w, start_grads_b = start_grads

        all_spectrum_mitx = []
        all_spectrum_mity = []
        all_spectrum_iter = []
        #feed_dict = {model.input:full_input, model.target:full_target, model.lr:lr}
        for level_ind, noise_l_list in enumerate(noise_std_klayers_llevels):
            print ("evaluating noise level "+str(level_ind))

            mi_ts = []

            x_samples, y_samples = get_subsample_data(noisy_mi_sampling_num)
            #y_values = y_samples.data
            sample_num = len(x_samples)
            for sample_i in range(sample_num):
                if sample_i % 100 == 0:
                    print ("sampling (noisy_w, x) " + str(sample_i/float(sample_num)) + " completion")

                x_inp = np.expand_dims(trn.X[sample_i], axis=0)
                y_tgt = np.expand_dims(trn.Y_oh[sample_i], axis=0)
                assign_vars = []
                for layer_k, noise_l_level_k in enumerate(noise_l_list):
                    sp_w = start_ws[layer_k].shape
                    sp_b = start_bs[layer_k].shape
                    noise_l_level_k_w, noise_l_level_k_b = noise_l_level_k
                    noise_flat_w = np.random.normal(0, noise_l_level_k_w, sp_w[0]*sp_w[1])
                    noises_b = np.random.normal(0, noise_l_level_k_b, sp_b)
                    noises_w = noise_flat_w.reshape(sp_w)
                    w_noised = start_ws[layer_k] - lr * (start_grads_w[layer_k] + noises_w) # 
                    b_noised = start_bs[layer_k] - lr * (start_grads_b[layer_k] + noises_b) #
                    assign_vars.append((layer_k, w_noised, b_noised))

                model.assign_vars(assign_vars, session)

                t_values_sample_i_noise_l = session.run(model_sgd.ts, feed_dict={model_sgd.input:x_inp, model_sgd.target:y_tgt})
                mi_ts.append(t_values_sample_i_noise_l)

            mi_ts_tensors = []

            for layer_k in range(len(noise_l_list)):
                t_tensor = np.concatenate([x[layer_k] for x in mi_ts], axis=0)
                mi_ts_tensors.append(t_tensor)

            print ("calculating noisy mi")
            xts_level_l, yts_level_l, noise_iters = get_layers_MI(mi_ts_tensors, x_samples, y_samples, start_iter_num)
            all_spectrum_mitx.append(xts_level_l)
            all_spectrum_mity.append(yts_level_l)
            all_spectrum_iter.append(noise_iters)

        return all_spectrum_mitx, all_spectrum_mity, all_spectrum_iter



    last_iter = 0
    current_iter = 0

    d = sio.loadmat('train.mat')
    F = d['tr_x']
    y = d['tr_y']
    C = type('type_C', (object,), {})
    trn = C()
    trn.X = F
    trn.Y_oh = y
 
    for break_point in breaking_points:

        iter_loop_num = break_point - last_iter

        bgd_imx, bgd_imy, iter_nums = bgd_trajectory(current_iter, iter_loop_num)
        #bgd_imx, bgd_imy, iter_nums = bgd_trajectory(current_iter, 2000)
        #bgd_imx_f = open('bgd_imx','wb')
        #pickle.dump(bgd_imx,bgd_imx_f)
        #bgd_imy_f = open('bgd_imy','wb')
        #pickle.dump(bgd_imy,bgd_imy_f)
        
        #bgd_imx_f = open('bgd_imx','rb')
        #read_imx = pickle.load(bgd_imx_f)
        #bgd_imy_f = open('bgd_imy','rb')
        #read_imy = pickle.load(bgd_imy_f)
        
        print ("bgd update completed at" + str(current_iter))
        current_iter += iter_loop_num
        last_iter = break_point

        lr = get_lr(current_iter)
        # snapshot of w recorded
        full_input, full_target = get_full_data2()
        feed_dict = {model.input:full_input, model.target:full_target, model.lr:lr}

        freeze_eval_list = [x for x in model.tvars]
        freeze_eval_list2 = [x for x in model.grads]
        freeze_eval_list.extend(freeze_eval_list2)
        freeze_tensors = session.run(freeze_eval_list, feed_dict=feed_dict)
        freeze_ws = freeze_tensors[:len(model.ws)]
        freeze_bs = freeze_tensors[len(model.ws):len(model.ws)*2]
        freeze_tvars = (freeze_ws, freeze_bs)
        freeze_grads_w = freeze_tensors[len(model.ws)*2:len(model.ws)*3]
        freeze_grads_b = freeze_tensors[len(model.ws)*3:]
        freeze_grads = (freeze_grads_w, freeze_grads_b)
        mean_signal_amplitude_w = [np.mean(np.abs(x)) for x in freeze_grads_w]
        mean_signal_amplitude_b = [np.mean(np.abs(x)) for x in freeze_grads_b]
        current_iter += 1

        noise_std_ks_levels = []
        for noise_ind, noise_l in enumerate(noise_levels):
            noise_std_ks_levels.append([])
            for layer_k in range(len(mean_signal_amplitude_w)):
                noise_std_w = mean_signal_amplitude_w[layer_k] * noise_l
                noise_std_b = mean_signal_amplitude_b[layer_k] * noise_l
                noise_std_ks_levels[noise_ind].append((noise_std_w, noise_std_b))

        zero_noise_imx, zero_noise_imy, zero_noise_iter = zero_noise_trajectory_point(freeze_tvars, freeze_grads, current_iter, lr)
        print ("one step zero noise point completed")

        # foreach noise level calculate its one-iteration mi by MC-sampling
        layer_noised_imx, layer_noised_imy, layer_noised_iter = noised_trajectory_point(freeze_tvars, freeze_grads, noise_std_ks_levels, current_iter, lr)
        print ("one step noise levels points completed")



collect_trajectory(bgd_stairs, noise_levels)