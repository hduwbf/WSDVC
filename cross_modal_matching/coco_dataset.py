import json
from collections import OrderedDict
from pathlib import Path
from typing import List
import csv
import h5py
import numpy as np
import torch
import torch.utils.data as data
from easydict import EasyDict
from tqdm import tqdm
import random
import utils
from text_embedding import *

class COCOImgFeatureLoader:
    def __init__(self, img_feature_path, keys):
        img_features = h5py.File(img_feature_path, 'r')
        self.keys = keys
        self.image_data = {}
        for key in tqdm(self.keys, desc='preload image'):
            img_feature = torch.Tensor(img_features[key]['features'])
            self.image_data[key] = img_feature
        img_features.close()
    def __getitem__(self, key):
        return self.image_data[key]

class COCOCapFeatureLoader:
    def __init__(self, caption_feature_path, keys):
        caption_features = h5py.File(caption_feature_path, 'r')
        self.keys = keys
        self.caption_data = {}
        for key in tqdm(self.keys, desc='preload caption'):
            caption_feature = torch.Tensor(caption_features[key]['features'])
            self.caption_data[key] = caption_feature
        caption_features.close()
    def __getitem__(self, key):
        return self.caption_data[key]

class ImgDatasetFeatures(data.Dataset):
    def __init__(self, img_feature_path, caption_feature_path, caption_path):
        img_features = h5py.File(img_feature_path, 'r')
        caption_features = h5py.File(caption_feature_path, 'r')
        self.keys = list(set(img_features.keys()).intersection(list(caption_features.keys())))
        img_features.close()
        caption_features.close()
        print(f'init dataset coco length {len(self)}')
        self.img_features = COCOImgFeatureLoader(img_feature_path, self.keys)
        self.caption_features = COCOCapFeatureLoader(caption_feature_path, self.keys)
        self.preproc_par_fn = preprocess_bert_paragraph
        self.caption_file = json.load(open(caption_path, 'r'))
        self.is_train = True
    def get_frames_from_video(self, vid_data, indices=None, num_frames=None):
        vid_len = vid_data.shape[0]
        if num_frames is not None:
            indices = utils.compute_indices(
                vid_len, num_frames, self.is_train)
        frames = vid_data[indices]
        return frames
    def get_frames_from_segment(
            self, vid_data, seg_num, clip_num, num_frames):
        start_frame = seg_num*num_frames
        seg_len = num_frames
        indices = utils.compute_indices(seg_len, num_frames, self.is_train)
        indices += start_frame
        frames = self.get_frames_from_video(vid_data, indices)
        return frames
    def get_pseudo_video(self, _id):
        caption = self.caption_file[_id]
        img_feature = torch.Tensor(self.img_features[_id])
        vid_len = random.randint(20, 99)
        vid_feature = torch.zeros(vid_len, img_feature.shape[1])
        vid_feature[:vid_len] = img_feature
        vid_feature += (0.1**0.5)*torch.randn_like(vid_feature)
        return vid_len, vid_feature

    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        cap_ids = []
        cap_ids.append(self.keys[index])
        clip_num = 4
        for i in range(clip_num-1):
            cap_ids.append(random.choices(self.keys, k=1)[0])
        captions = []
        for cap_id in cap_ids:
            captions.append(self.caption_file[cap_id])

        # load video frames
        img_lens = []
        img_features = []
        for cap_id in cap_ids:
            img_len, img_feature = self.get_pseudo_video(cap_id)
            img_lens.append(img_len)
            img_features.append(img_feature)
        vid_data = torch.cat(img_features, dim=0)
        vid_frames_len = vid_data.shape[0]
        if vid_frames_len>80:
            vid_frames_len = 80
        vid_frames = self.get_frames_from_video(vid_data, num_frames=vid_frames_len)

        # load segment frames
        clip_frames_list = []
        clip_frames_len_list = []
        for i in range(len(img_lens)):
            c_num_frames = img_lens[i]
            if c_num_frames>80:
                c_num_frames = 80
            c_frames = self.get_frames_from_video(img_features[i], num_frames=c_num_frames)
            clip_frames_list.append(c_frames)
            clip_frames_len_list.append(c_frames.shape[0])

        # load text
        list_of_list_of_words = self.preproc_par_fn(captions)

        # load precomputed text features
        cap_features = []
        cap_lens = []
        for cap_id in cap_ids:
            cap_feature = torch.Tensor(self.caption_features[cap_id])
            cap_features.append(cap_feature)
            cap_lens.append(int(cap_feature.shape[0]))
        par_cap_vectors = torch.cat(cap_features, dim=0)
        sent_cap_len_list = cap_lens
        par_cap_len = int(par_cap_vectors.shape[0])
        par_cap_vectors = par_cap_vectors.float()

        # split paragraph features into sentences
        sent_cap_vectors_list = []
        pointer = 0
        for i, sent_cap_len in enumerate(sent_cap_len_list):
            sent_cap_vectors = par_cap_vectors[
                               pointer:pointer + sent_cap_len, :]
            sent_cap_vectors_list.append(sent_cap_vectors)
            pointer += sent_cap_len
        return {
            "vid_id": cap_id, #video name
            "data_words": list_of_list_of_words, #sentences word
            "vid_frames": vid_frames, #video features
            "vid_frames_len": vid_frames_len, #video features len
            "par_cap_vectors": par_cap_vectors, #paragraph features
            "par_cap_len": par_cap_len, #paragraph features len
            "clip_num": clip_num,
            "sent_num": clip_num,
            "clip_frames_list": clip_frames_list,
            "clip_frames_len_list": clip_frames_len_list,
            "sent_cap_len_list": sent_cap_len_list,
            "sent_cap_vectors_list": sent_cap_vectors_list
        }

    def collate_fn(self, data_batch):
        def get_data(key):
            return [d[key] for d in data_batch]

        batch_size = len(data_batch)

        # collate video frames
        list_vid_frames = get_data("vid_frames")
        list_vid_frames_len = get_data("vid_frames_len")
        vid_feature_dim = list_vid_frames[0].shape[-1]
        vid_frames_len = torch.tensor(list_vid_frames_len).long()
        vid_frames_max_seq_len = int(vid_frames_len.max().numpy())
        vid_frames = torch.zeros(
            batch_size, vid_frames_max_seq_len, vid_feature_dim).float()
        vid_frames_mask = torch.zeros(batch_size, vid_frames_max_seq_len)
        for batch, (seq_len, item) in enumerate(zip(
                list_vid_frames_len, list_vid_frames)):
            vid_frames[batch, :seq_len] = item
            vid_frames_mask[batch, :seq_len] = 1

        # collate paragraph features
        list_par_cap_len = get_data("par_cap_len")
        list_par_cap_vectors = get_data("par_cap_vectors")
        par_feature_dim = list_par_cap_vectors[0].shape[-1]
        par_cap_len = torch.tensor(list_par_cap_len).long()
        par_cap_max_len = int(par_cap_len.max().numpy())
        par_cap_vectors = torch.zeros(
            batch_size, par_cap_max_len, par_feature_dim).float()
        par_cap_mask = torch.zeros(batch_size, par_cap_max_len)
        for batch, (seq_len, item) in enumerate(
                zip(list_par_cap_len, list_par_cap_vectors)):
            par_cap_vectors[batch, :seq_len, :] = item
            par_cap_mask[batch, :seq_len] = 1

        # collate clip frames
        list_clip_num = get_data("clip_num")
        clip_num = torch.tensor(list_clip_num).long()
        total_clip_num = int(np.sum(list_clip_num))
        list_clip_frames_len_list = get_data("clip_frames_len_list")
        clip_frames_max_len = int(np.max(
            [np.max(len_single) for len_single in list_clip_frames_len_list]))
        clip_frames = torch.zeros((
            total_clip_num, clip_frames_max_len, vid_feature_dim)).float()
        clip_frames_mask = torch.zeros(
            (total_clip_num, clip_frames_max_len))
        list_clip_frames_list = get_data("clip_frames_list")
        clip_frames_len = []
        c_num = 0
        for batch, clip_frames_list in enumerate(list_clip_frames_list):
            for i, clip_frames_item in enumerate(clip_frames_list):
                clip_frames_len_item = int(clip_frames_item.shape[0])
                clip_frames[c_num, :clip_frames_len_item, :] =\
                    clip_frames_item
                clip_frames_mask[c_num, :clip_frames_len_item] = 1
                clip_frames_len.append(clip_frames_len_item)
                c_num += 1
        clip_frames_len = torch.tensor(clip_frames_len).long()

        # collate sentence features
        list_sent_num = get_data("sent_num")
        sent_num = torch.tensor(list_sent_num).long()
        total_sent_num = int(np.sum(list_sent_num))
        list_sent_cap_len_list = get_data("sent_cap_len_list")
        sent_cap_max_len = int(np.max(
            [np.max(len_single) for len_single in list_sent_cap_len_list]))
        sent_cap_len = []
        sent_cap_mask = torch.zeros(
            (total_sent_num, sent_cap_max_len)).long()
        cap_feature_dim = list_par_cap_vectors[0].shape[-1]
        sent_cap_vectors = torch.zeros(
            (total_sent_num, sent_cap_max_len, cap_feature_dim))
        c_num = 0
        for batch, sent_cap_len_list in enumerate(
                list_sent_cap_len_list):
            pointer = 0
            for sent_cap_len_item in sent_cap_len_list:
                sent_cap_vectors[c_num, :sent_cap_len_item] =\
                    par_cap_vectors[
                    batch, pointer:pointer + sent_cap_len_item]
                sent_cap_mask[c_num, :sent_cap_len_item] = 1
                sent_cap_len.append(sent_cap_len_item)
                c_num += 1
                pointer += sent_cap_len_item
        sent_cap_len = torch.tensor(sent_cap_len).long()

        return {
            "vid_frames": vid_frames,
            "vid_frames_mask": vid_frames_mask,
            "vid_frames_len": vid_frames_len,
            "par_cap_vectors": par_cap_vectors,
            "par_cap_mask": par_cap_mask,
            "par_cap_len": par_cap_len,
            "clip_num": clip_num,
            "clip_frames": clip_frames,
            "clip_frames_len": clip_frames_len,
            "clip_frames_mask": clip_frames_mask,
            "sent_num": sent_num,
            "sent_cap_vectors": sent_cap_vectors,
            "sent_cap_mask": sent_cap_mask,
            "sent_cap_len": sent_cap_len,
            "vid_id": get_data("vid_id"),
            "data_words": get_data("data_words")
        }

if __name__ == '__main__':
    from tqdm import tqdm
    dataset = ImgDatasetFeatures('../data/cocoFeatures.hdf5', '../data/coco_text_features.hdf5', '../data/coco_text.json')
    dataloader = data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=1, collate_fn=dataset.collate_fn,
        pin_memory=True)
    for batch_data in tqdm(dataloader):
        for dt in batch_data:
            print(batch_data[dt])
        break
        continue