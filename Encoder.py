from scipy.io import loadmat
from itertools import groupby
import numpy as np
import math
import re

behavior_names = ['chase', 'reorient', 'search', 'sing', 'stay close', 'still', 'touch', 'walk', 'other']


def behavior_decoder(hex,out_format = 'name'):
    bin = format(int(hex, 16), "08b")
    ind_ls = []
    if bin != '00000000':
        for m in re.finditer('1', bin):
            ind_ls.append(m.start())
    else:
        ind_ls = [8]
    if out_format == 'name':
        behavior = [behavior_names[ind] for ind in ind_ls]
        return behavior
    elif out_format == 'ind':
        return ind_ls


def behavior_code_gen(behavior_ind, overlap=0):
    if overlap:
        temp_ls = [format(x, '07b') for x in range(int(math.pow(2, 7)))]
        ls = [x[:behavior_ind] + '1' + x[behavior_ind:] for x in temp_ls]
        hex_ls = [format(int(x, 2), "02X") for x in ls]
    else:
        temp_ls = '0000000'
        ls = temp_ls[:behavior_ind] + '1' + temp_ls[behavior_ind:]
        hex_ls = format(int(ls, 2), "02X")
    return hex_ls


def seq_gen():
    condition_num = 2
    video_num = 19
    behavior_num = 8

    temp_data = loadmat('data.mat')
    data = [0] * condition_num

    data_no_repeat = [0] * condition_num
    data_no_repeat_str = [0] * condition_num
    data_no_repeat_len = [0] * condition_num
    data_no_repeat_cum_len = [0] * condition_num

    for condition in range(condition_num):
        data[condition] = [0] * video_num
        data_no_repeat[condition] = [0] * video_num
        data_no_repeat_str[condition] = [0] * video_num
        data_no_repeat_len[condition] = [0] * video_num
        data_no_repeat_cum_len[condition] = [0] * video_num
        for video in range(video_num):
            data[condition][video] = [0] * temp_data['video_data'][condition][0][0][video].shape[1]
            for frame_ind in range(temp_data['video_data'][condition][0][0][video].shape[1]):
                temp_str = "".join(str(x) for x in temp_data['video_data'][condition][0][0][video][
                    range(behavior_num), frame_ind].tolist())
                data[condition][video][frame_ind] = format(int(temp_str, 2), "02X")
            run_length = [[val, len([*thing])] for val, thing in groupby(data[condition][video])]
            data_no_repeat[condition][video] = [x[0] for x in run_length]
            data_no_repeat_len[condition][video] = [x[1] for x in run_length]
            data_no_repeat_cum_len[condition][video] = data_no_repeat_len[condition][video]
            data_no_repeat_cum_len[condition][video].insert(0, 1)
            data_no_repeat_cum_len[condition][video] = np.cumsum(data_no_repeat_cum_len[condition][video])
            data_no_repeat_str[condition][video] = ' '.join([str(x) for x in data_no_repeat[condition][video]])
    output = {'ls': data_no_repeat, 'len': data_no_repeat_len,'cum_len': data_no_repeat_cum_len, 'str': data_no_repeat_str}
    return output
