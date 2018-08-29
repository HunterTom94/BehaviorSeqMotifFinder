# for behavioral transition matrix

import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import motmot.fview as fview
import sys
import numpy as np
import scipy.io as sio


def clip_gen(clips_log, clip_ind=[]):
    # clips_log = r'Z:\April\JAABA\Experiments\CS\transition_clips\clips.mat'
    if 'WT' in clips_log:
        dirc_log = r'Z:\April\JAABA\Experiments\test\WT_dirc.mat'
        mat_contents = sio.loadmat(clips_log)
        dirc_contents = sio.loadmat(dirc_log)
        log = mat_contents['WT']
        index = mat_contents['WT_index']
        dirc = dirc_contents['WT_dirc']
    elif 'Or47b' in clips_log:
        dirc_log = r'Z:\April\JAABA\Experiments\test\or47b_dirc.mat'
        mat_contents = sio.loadmat(clips_log)
        dirc_contents = sio.loadmat(dirc_log)
        log = mat_contents['Or47b']
        index = mat_contents['Or47b_index']
        dirc = dirc_contents['or47b_dirc']

    if not clip_ind:
        clip_no = range(len(log))
    elif clip_ind:
        clip_no = clip_ind
    video_no = len(dirc[0])

    for j in clip_no:
        if 'WT' in clips_log:
            fname_2 = 'Z:\April\JAABA\Experiments\CS\\transition_clips\WT\\' + ''.join(log[j, 1]) + '.fmf'
        elif 'Or47b' in clips_log:
            fname_2 = 'Z:\April\JAABA\Experiments\CS\\transition_clips\or47b\\' + ''.join(log[j, 1]) + '.fmf'

        fmf_saver = FMF.FlyMovieSaver(fname_2)  # save each video based on clip name

        b = log[j, 0]  # get the log of this clip
        segment_no = index[j]
        timestamp_count = 0

        for i in range(video_no):
            a = dirc[0, i][0]  # a = dirc[row 0, video_no][get string of video name]

            if 'WT' in clips_log:
                fname = 'Z:\April\JAABA\Experiments\CS\WT\\' + a + '\\video.fmf'
            elif 'Or47b' in clips_log:
                fname = 'Z:\April\JAABA\Experiments\CS\or47b\\' + a + '\\video.fmf'

            fmf = FMF.FlyMovie(fname)

            c = b[i, :]
            segment_length = np.array(segment_no).tolist()
            segment_length = segment_length[0]

            for k in range(int(segment_length[i])):  # for the num of segments in each video, -1 due to python index
                if int(segment_length[i]) == 0:
                    e = 1
                else:
                    e = c[k]
                    f = np.array(e).tolist()
                    d = f[0]

                for frame_number in d:
                    frame, timestamp = fmf.get_frame(int(frame_number) - 1)
                    timestamp_count += 1
                    fmf_saver.add_frame(frame, timestamp_count)

                # add the spacer b/w segments
                width, height = 1024, 1024
                fake_image = np.zeros((height, width), dtype=np.uint8)
                for p in range(10):
                    timestamp_count += 1
                    fmf_saver.add_frame(fake_image, timestamp_count)

            # add the spacer b/w videos
            width, height = 1024, 1024
            fake_image = 255 * np.ones((height, width), dtype=np.uint8)
            for p in range(10):
                timestamp_count += 1
                fmf_saver.add_frame(fake_image, timestamp_count)

        fmf_saver.close()
