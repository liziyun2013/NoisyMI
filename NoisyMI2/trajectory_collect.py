# -*- coding:utf-8 -*-
__author__ = 'st491'
from model.network import SimpleNN
import tensorflow as tf
from data.DataHelper import trn, get_full_data, get_subsample_data
import numpy as np

session = tf.Session()
default_g = tf.Graph().as_default()
session.as_default()

# 网络结构声明
model = SimpleNN([[784, 128], [128, 128], [128, 128], [128, 128], [128, 10]], 60000)
model_sgd = SimpleNN([[784, 128], [128, 128], [128, 128], [128, 128], [128, 10]], 1, reuse=True)

# 噪音信号比等级声明
noise_levels = [0.05, 0.2, 0.5, 1.0, 2.0]

# 采样(w, x)计算带噪音的t的采样数（最好等同于Mnist数据大小=60000）
noisy_mi_sampling_num = 200

# 单轮噪音引入所在的bgd迭代轮数
bgd_stairs = [100, 300, 500]

# lr小于0时，bgd使用adam优化器，大于0时使用gradient descent优化器。tuple=(小于该轮数时, 使用的lr值)
lr_adjust = [(100, -0.05), (5000, 0.5)]

# 是否真正计算MI
enable_mi_calculation = False

tf.global_variables_initializer().run(session=session)

def get_MI(t_val, x_val, y_val):
    if enable_mi_calculation:
        return 0.0, 0.0
    else:
        return 0.0, 0.0

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

        for i in range(run_iter):
            lr = get_lr(i+start_iter_num)

            full_inp, full_tgt = get_full_data()
            feed_dict = {model.input:full_inp, model.target:full_tgt, model.lr:lr}

            eval_list = [x for x in model.ts]
            eval_list.append(model.cost)
            ts_op = model.sess_run_op(session, eval_list, feed_dict, lr)
            ts_tensor = ts_op[:-2]
            cost = ts_op[-2]

            xts, yts, _ = get_layers_MI(ts_tensor, full_inp, full_tgt, start_iter_num+i)

            mi_tx.append(xts)
            mi_ty.append(yts)
            iters.append(start_iter_num + i)

            if i % 10 == 0:
                print (i,cost)

        return mi_tx, mi_ty, iters

    def zero_noise_trajectory_point(start_vars, start_var_grads, start_iter_num, lr):
        start_ws, start_bs = start_vars
        start_grads_w, start_grads_b = start_var_grads

        feed_dict = {model.input:full_input, model.target:full_target, model.lr:lr}
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

        for level_ind, noise_l_list in enumerate(noise_std_klayers_llevels):
            print ("evaluating noise level "+str(level_ind))

            mi_ts = []

            x_samples, y_samples = get_subsample_data(noisy_mi_sampling_num)

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
                    w_noised = start_ws[layer_k] - lr * (start_grads_w[layer_k] + noises_w)
                    b_noised = start_bs[layer_k] - lr * (start_grads_b[layer_k] + noises_b)
                    assign_vars.append((layer_k, w_noised, b_noised))

                model_sgd.assign_vars(assign_vars, session)

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
    for break_point in breaking_points:

        iter_loop_num = break_point - last_iter

        bgd_imx, bgd_imy, iter_nums = bgd_trajectory(current_iter, iter_loop_num)
        print ("bgd update completed at" + str(current_iter))
        current_iter += iter_loop_num
        last_iter = break_point

        lr = get_lr(current_iter)
        # snapshot of w recorded
        full_input, full_target = get_full_data()
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
        current_iter +=1

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