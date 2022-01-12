import scipy.io as sio
import math

from base_code.basisModel import display_stats as base_display_stats

# cifar10_min_acc = {'vgg16':90.0, 'resnet56':92.0}
# cifar100_min_acc = {'vgg16':61.0, 'resnet110':71.0}

def display_stats(basis_model, model, exp_name, input_size=[32, 32], count_relu=False):
    return base_display_stats(basis_model, model, exp_name, input_size, count_relu)

def get_t_vec(model_name, min_map, display=True):

    mat = sio.loadmat(model_name+'-layerwise_results.mat')
    results = mat['results']
    t_vec = []
    t_list = [1.0, 0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            if results[i, j] < min_map:
                t_vec.append(t_list[j])
                break
            elif j == results.shape[1]-1:
                t_vec.append(t_list[-1])

    if display:
        print('\n########################################### Compression Stats ###########################################\n')
        print('    Base Accuracy: %.5f' % mat['base_accuracy'].item())
        print('    Min Accuracy: %.5f' % min_map)
        print('    t_vec: ', t_vec)
        print('\n#########################################################################################################\n')

    return t_vec

def get_Q_vec(model_name, filter_list, min_map, min_Q=4, display=True):

    mat = sio.loadmat('layerwise/' + model_name+'-layerwise_results_f.mat')
    results = mat['results']
    Q_vec = []
    t_list = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            if results[i, j] < min_map:
                # c_f = math.ceil(filter_list[i] * t_list[j])
                c_f = max(math.ceil(filter_list[i] * t_list[j]), min_Q)
                Q_vec.append(c_f)
                break
            elif j == results.shape[1]-1:
                c_f = max(math.ceil(filter_list[i] * t_list[-1]), min_Q)
                # c_f = math.ceil(filter_list[i] * t_list[-1])
                Q_vec.append(c_f)

    if display:
        print('\n########################################### Compression Stats ###########################################\n')
        print('    Base Accuracy: %.5f' % mat['base_accuracy'].item())
        print('    Min Accuracy: %.5f' % min_map)
        print('    Q_vec: ', Q_vec)
        print('\n#########################################################################################################\n')

    return Q_vec