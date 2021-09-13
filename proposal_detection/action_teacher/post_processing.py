# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import multiprocessing as mp
import opts
opt = opts.parse_opt()
opt = vars(opt)
from utils import iou_with_anchors


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def getDatasetDict(opt):
    # TODO:ActivityNet
    anno_path = opt["anno_path"] + 'new_th14.json'
    list_path = opt["anno_path"] + 'validation_list.json'
    database = load_json(anno_path)


    video_list = load_json(list_path)
    video_dict = {}
    for i in range(len(video_list)):
        video_name = video_list[i]
        video_info = database[video_name]
        video_new_info = {}

        # TODO:ActivityNet
        video_new_info['duration_second'] = video_info['duration_second']
        video_new_info['annotations'] = video_info['annotations']

        video_dict[video_name] = video_new_info
    return video_dict


def soft_nms(df, alpha, t1, t2):
    '''
    df: proposals generated by network;
    alpha: alpha value of Gaussian decaying function;
    t1, t2: threshold for soft nms.
    '''
    df = df.sort_values(by="score", ascending=False)
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 1 and len(rscore) < 101:
        max_index = tscore.index(max(tscore))
        tmp_iou_list = iou_with_anchors(
            np.array(tstart),
            np.array(tend), tstart[max_index], tend[max_index])
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = tmp_iou_list[idx]
                tmp_width = tend[max_index] - tstart[max_index]
                if tmp_iou > t1 + (t2 - t1) * tmp_width:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) /
                                                       alpha)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def video_post_process(opt, video_list, video_dict):
    for video_name in video_list:
        df = pd.read_csv("./output/BMN_results/" + video_name + ".csv")

        if len(df) > 1:
            snms_alpha = opt["soft_nms_alpha"]
            snms_t1 = opt["soft_nms_low_thres"]
            snms_t2 = opt["soft_nms_high_thres"]
            df = soft_nms(df, snms_alpha, snms_t1, snms_t2)

        df = df.sort_values(by="score", ascending=False)
        video_info = video_dict[video_name]
        duration_second = video_info['duration_second']
        proposal_list = []

        for j in range(min(100, len(df))):
            tmp_proposal = {}
            tmp_proposal["score"] = df.score.values[j]
            tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * duration_second,
                                       min(1, df.xmax.values[j]) * duration_second]
            proposal_list.append(tmp_proposal)
        result_dict[video_name] = proposal_list


def BMN_post_processing(opt):
    video_dict = getDatasetDict(opt)
    video_list = list(video_dict.keys())  # [:100]
    global result_dict
    result_dict = mp.Manager().dict()

    num_videos = len(video_list)
    num_videos_per_thread = num_videos // opt["post_process_thread"]
    processes = []
    for tid in range(opt["post_process_thread"] - 1):
        tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) * num_videos_per_thread]
        p = mp.Process(target=video_post_process, args=(opt, tmp_video_list, video_dict))
        p.start()
        processes.append(p)
    tmp_video_list = video_list[(opt["post_process_thread"] - 1) * num_videos_per_thread:]
    p = mp.Process(target=video_post_process, args=(opt, tmp_video_list, video_dict))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"], "w")
    json.dump(output_dict, outfile)
    outfile.close()

def random_post_processing(opt):
    video_dict = getDatasetDict(opt)
    video_list = list(video_dict.keys())  # [:100]
    result_dict = {}
    for video_name in video_list:
        result = []
        duration = video_dict[video_name]['duration']
        for i in range(100):
            anno = {}
            anno["score"] = random.random()
            start = max(0, random.random())
            end = min(1, start+random.random())
            anno["segment"] = [start*duration, end*duration]
            result.append(anno)
        result_dict[video_name] = result
    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open('./output/random_proposal.json', "w")
    json.dump(output_dict, outfile)
    outfile.close()

if __name__ == '__main__':
    import random
    random_post_processing(opt)