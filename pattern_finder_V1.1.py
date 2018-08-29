from Encoder import seq_gen, behavior_code_gen, behavior_decoder
from ClipCutter import clip_gen
import itertools
import time
import pickle
import re
import numpy as np
import scipy.io as spio
import pandas as pd
from openpyxl import load_workbook

condition_name = ['Or47b', 'WT']
behavior_names = ['chase', 'reorient', 'search', 'sing', 'stay close', 'still', 'touch', 'walk', 'other']


def segment_finder(seq, behavior_code, mode='loop', length=2):
    behavior_code_ind = [i for i, x in enumerate(seq) if x in behavior_code]
    if mode == 'loop':
        segment = []
        for ind in range(len(behavior_code_ind) - 1):
            temp_segment = seq[behavior_code_ind[ind]:(behavior_code_ind[ind + 1] + 1)]
            if temp_segment[0] == temp_segment[-1]:
                segment.append(temp_segment)
    elif mode == 'open':
        segment = []
        for ind in range(len(behavior_code_ind)):
            temp_segment = seq[behavior_code_ind[ind]:(behavior_code_ind[ind] + length)]
            if len(temp_segment) == length:
                segment.append(temp_segment)
    elif mode == 'to_new':
        segment = []
        end_ind = 0
        for ind in range(len(behavior_code_ind)):
            start_ind = behavior_code_ind[ind]
            if end_ind >= start_ind:
                continue
            else:
                while seq[start_ind] in behavior_code_gen(behavior_decoder(behavior_code, out_format='ind')[0],
                                                          overlap=1):
                    end_ind = start_ind + 1
                    start_ind += 1
                segment.append(seq[behavior_code_ind[ind]:end_ind + 1])
    return segment


def unique_segment(s):
    n = len(s)
    try:
        t = list(s)
        t.sort()
    except TypeError:
        del t
    else:
        assert n > 0
        last = t[0]
        lasti = i = 1
        count = []
        cnt = 1
        while i < n:
            if t[i] != last:
                t[lasti] = last = t[i]
                lasti += 1
                count.append(cnt)
                cnt = 1
            else:
                cnt += 1
            if i == n - 1:
                count.append(cnt)
            i += 1
        t_to_take = [0] * len(t[:lasti])
        for j in range(len(t[:lasti])):
            t_to_take[j] = [behavior_decoder(x, out_format='name') for x in t[:lasti][j]]
        output = [0] * len(count)
        for ind in range(len(count)):
            output[ind] = [t_to_take[ind], count[ind], ' '.join(t[:lasti][ind])]
        output.sort(reverse=True, key=lambda x: x[1])
        return output


def to_new_pattern(seq, condition=0, behavior_ind=1, behavior_overlap=0):
    pattern_mode = 'to_new'
    data = seq['ls']
    segment = [0] * len(data)
    condition_segment = [0] * len(data)
    segment[condition] = []
    for video in range(len(data[condition])):
        segment[condition].append(
            segment_finder(data[condition][video], behavior_code_gen(behavior_ind, overlap=behavior_overlap),
                           mode=pattern_mode))
    condition_segment[condition] = list(itertools.chain.from_iterable(segment[condition]))  # join inner lists
    unique_segment_ls = unique_segment(condition_segment[condition])
    start_end = []
    whole_pattern = []
    num_repeats = []
    for _, item in enumerate(unique_segment_ls):
        start_end.append([item[0][0], item[0][-1]])
        whole_pattern.append(item[0])
        num_repeats.append(item[1])

    df = pd.DataFrame.from_dict({'start_end': start_end, 'whole pattern': whole_pattern, 'num_repeats': num_repeats})

    path = '%s.xlsx' % (condition_name[condition])
    book = load_workbook(path)
    writer = pd.ExcelWriter(path, engine='openpyxl')
    writer.book = book

    df.to_excel(writer, behavior_names[behavior_ind], index=False)
    writer.save()
    writer.close()


def pattern_clips(seq, condition=0, behavior_ind=1, behavior_overlap=0, pattern_mode='open', pattern_length=2,
                  save=0):
    if behavior_overlap:
        assert pattern_mode != 'to_new', 'segment start has multiple behavior'
    data = seq['ls']
    data_len = seq['len']
    data_cum_len = seq['cum_len']
    data_str = seq['str']
    segment = [0] * len(data)
    condition_segment = [0] * len(data)
    segment[condition] = []
    for video in range(len(data[condition])):
        segment[condition].append(
            segment_finder(data[condition][video], behavior_code_gen(behavior_ind, overlap=behavior_overlap),
                           mode=pattern_mode, length=pattern_length))
    condition_segment[condition] = list(itertools.chain.from_iterable(segment[condition]))  # join inner lists
    unique_segment_ls = unique_segment(condition_segment[condition])
    min_repeats_to_show = 2
    unique_segment_obj_arr = np.zeros((len([x for x in unique_segment_ls if x[1] >= min_repeats_to_show]), 3),
                                      dtype=np.object)
    clip_num = np.zeros((unique_segment_obj_arr.shape[0], 1), dtype=np.object)
    idx = 0
    for _, item in enumerate(unique_segment_ls):
        if pattern_mode == 'to_new':
            print([item[0][0], item[0][-1]])
            print(item[0])
            print(item[1])

        else:
            print(item[0:2])
        if item[1] < min_repeats_to_show:
            continue
        segment_piece_length = np.zeros((item[1], len(item[0])))
        repeat_ind = 0
        pattern_sum = 0
        video_max_repeat = 0
        clip_num[idx, 0] = np.zeros((len(data[condition]), 1), dtype=np.object)
        for video in range(len(data[condition])):
            m_start = [m.start() for m in re.finditer(item[2], data_str[condition][video])]
            clip_num[idx, 0][video, 0] = len(m_start)
            pattern_sum += len(m_start)
            video_max_repeat = max(video_max_repeat, len(m_start))
        video_obj_arr = np.zeros((len(data[condition]), video_max_repeat), dtype=np.object)
        for video in range(len(data[condition])):
            m_start = [m.start() for m in re.finditer(item[2], data_str[condition][video])]
            m_end = [m.end() for m in re.finditer(item[2], data_str[condition][video])]
            if m_start:
                for m_ind in range(len(m_start)):
                    for segment_piece_ind in range(len(item[0])):
                        segment_piece_length[repeat_ind,segment_piece_ind] = data_len[condition][video][int(m_start[m_ind]/3 + segment_piece_ind)]
                    repeat_ind += 1
                    pattern_length = ((m_end[m_ind] + 1) - m_start[m_ind]) / 3
                    clip_start = data_cum_len[condition][video][int(m_start[m_ind] / 3)]  # matlab index
                    clip_end = data_cum_len[condition][video][int(m_start[m_ind] / 3 + pattern_length)]  # matlab index
                    video_obj_arr[video, m_ind] = np.arange(clip_start, clip_end)
        print([int(round(x)) for x in np.mean(segment_piece_length, axis=0)])
        unique_segment_obj_arr[idx, 0] = video_obj_arr
        unique_segment_obj_arr[idx, 1] = ' '.join(str(x) for x in item[0])
        unique_segment_obj_arr[idx, 2] = pattern_sum
        idx += 1

    overlap_names = ['nonoverlap', 'overlap']
    print()
    if save:
        save_name = 'clip_index/%s_%s_%s_%s_%s_Pattern_clips.mat' % (
            condition_name[condition], behavior_names[behavior_ind], overlap_names[behavior_overlap], pattern_mode,
            str(int(pattern_length)))
        spio.savemat(save_name, mdict={'%s' % (condition_name[condition]): unique_segment_obj_arr,
                                       '%s_index' % (condition_name[condition]): clip_num})
        return save_name
    return []


if __name__ == "__main__":
    # seq = seq_gen()
    # with open('saved_seq', 'wb') as f:
    #     pickle.dump(seq, f)
    # exit()
    start = time.time()
    save = 1
    clip = 0
    with open('saved_seq', 'rb') as f:
        sequence = pickle.load(f)
    # for i in range(8):
    #     to_new_pattern(sequence, condition=1, behavior_ind=i, behavior_overlap=0)
    # exit()
    clip_log = pattern_clips(sequence, condition=1, behavior_ind=2, behavior_overlap=0, pattern_mode='open',
                             pattern_length=7, save=save)
    if save and clip:
        clip_gen(clip_log, [1, 2])
    end = time.time()
    print(end - start)
