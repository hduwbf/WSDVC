import json
from collections import OrderedDict
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
import torch.utils.data as data
from easydict import EasyDict
from tqdm import tqdm

import utils
from text_embedding import preprocess_bert_paragraph


class BertTextFeatureLoader:
    def __init__(self, dataset_path, ids, preload=True):
        self.h5_path = (dataset_path / "text_features.hdf5")
        lens_file = (dataset_path / f"text_lens.json")
        self.lens = json.load(open(lens_file, "r"))
        self.cached_data = None
        if preload:
            h5file = h5py.File(self.h5_path, "r")
            self.cached_data = {}
            for id_ in tqdm(ids, desc="preload text"):
                text_feature = torch.Tensor(h5file[id_]['features'])
                self.cached_data[id_] = text_feature
            h5file.close()

    def __getitem__(self, id_):
        lens = self.lens[id_]
        if self.cached_data is None:
            h5file = h5py.File(self.h5_path, "r")
            text_features = torch.Tensor(h5file[id_]['features'])
            h5file.close()
            return text_features, lens
        return self.cached_data[id_], lens


class ActivityNetVideoFeatureLoader:
    def __init__(self, dataset_path, ids, preload=True):
        self.dataset_path = Path(dataset_path)
        self.features_path = (dataset_path / "resnet152_features_activitynet_5fps_320x240.hdf5")
        self.cached_data = None
        if preload:
            h5file = h5py.File(self.features_path, 'r')
            self.cached_data = {}
            for id_ in tqdm(ids, desc="preload videos"):
                video_feature = torch.Tensor(h5file[id_])
                self.cached_data[id_] = video_feature

    def __getitem__(self, id_):
        if self.cached_data is None:
            h5file = h5py.File(self.features_path, 'r')
            video_feature = torch.Tensor(h5file[id_])
            h5file.close()
            return video_feature
        else:
            return self.cached_data[id_]

class VideoDatasetFeatures(data.Dataset):
    def __init__(
            self, dataset_name, dataset_path, dataset_features,
            split, max_frames, is_train,
            preload_vid_feat, preload_text_feat,
            frames_noise):
        self.frames_noise = frames_noise
        self.split = split
        self.max_frames = max_frames
        self.is_train = is_train
        json_file = dataset_path / f"{split}.json"
        self.vids_dict = json.load(open(json_file, "r"))
        self.ids = list(self.vids_dict.keys())
        print(f"init dataset {dataset_name} split {split} length {len(self)} ")
        self.text_data = BertTextFeatureLoader(
            dataset_path, self.ids, preload_text_feat)
        self.preproc_par_fn = preprocess_bert_paragraph
        self.vid_data = ActivityNetVideoFeatureLoader(
            dataset_path, self.ids, preload_vid_feat)

    def get_frames_from_video(self, vid_id, indices=None, num_frames=None):
        vid_dict = self.vids_dict[vid_id]
        vid_len = self.vid_data[vid_id].shape[0]
        if num_frames is not None:
            indices = utils.compute_indices(
                vid_len, num_frames, self.is_train)
        frames = self.vid_data[vid_id][indices]
        return frames

    def get_frames_from_segment(
            self, vid_id, seg_num, clip_num, num_frames):
        if self.is_train:
            start_frame = seg_num*int(self.vid_data[vid_id].shape[0]/clip_num)
            seg_len = int(self.vid_data[vid_id].shape[0]/clip_num)
            if seg_len == 0:
                seg_len = 1
            indices = utils.compute_indices(seg_len, num_frames, self.is_train)
            indices += start_frame
            frames = self.get_frames_from_video(vid_id, indices)
            return frames
            # start_frame = seg_num
            # seg_len = int(clip_num - seg_num) # end-start
            # if seg_len < 0:
            #     seg_len = -seg_len
            # elif seg_len == 0:
            #     seg_len = 1
            # indices = utils.compute_indices(seg_len, num_frames, self.is_train)
            # indices += start_frame
            # frames = self.get_frames_from_video(vid_id, indices)
            # return frames
        else:
            start_frame = seg_num
            seg_len = int(clip_num - seg_num) # end-start
            if seg_len == 0:
                seg_len = 1
            indices = utils.compute_indices(seg_len, num_frames, self.is_train)
            indices += start_frame
            frames = self.get_frames_from_video(vid_id, indices)
            return frames

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        vid_id = self.ids[index]
        vid_dict = self.vids_dict[vid_id]
        sent_num = len(vid_dict["sentences"])
        clip_num = sent_num

        # load video frames
        vid_frames_len = self.vid_data[vid_id].shape[0]
        if vid_frames_len > self.max_frames:
            vid_frames_len = self.max_frames
        vid_frames = self.get_frames_from_video(
            vid_id, num_frames=vid_frames_len)
        vid_frames_len = int(vid_frames.shape[0])
        if self.frames_noise != 0:
            vid_frames_noise = utils.truncated_normal_fill(
                vid_frames.shape, std=self.frames_noise)
            vid_frames += vid_frames_noise

        # load segment frames
        clip_frames_list = []
        clip_frames_len_list = []
        if self.is_train:
            for i in range(clip_num):
                c_num_frames = int(self.vid_data[vid_id].shape[0]/clip_num)
                if c_num_frames > self.max_frames:
                    c_num_frames = self.max_frames
                c_frames = self.get_frames_from_segment(
                    vid_id, i, clip_num, num_frames=c_num_frames)
                if self.frames_noise != 0:
                    clip_frames_noise = utils.truncated_normal_fill(
                        c_frames.shape, std=self.frames_noise)
                    c_frames += clip_frames_noise
                clip_frames_list.append(c_frames)
                clip_frames_len_list.append(c_frames.shape[0])
            # duration = vid_dict['duration']
            # vid_len = self.vid_data[vid_id].shape[0]
            # for i in range(clip_num):
            #     start_ = int(vid_dict['timestamps'][i][0]/duration*vid_len)
            #     end_ = int(vid_dict['timestamps'][i][1]/duration*vid_len)
            #     c_num_frames = end_-start_
            #     if c_num_frames < 0:
            #         c_num_frames = -c_num_frames
            #     elif c_num_frames == 0:
            #         c_num_frames = 1
            #     elif c_num_frames > self.max_frames:
            #         c_num_frames = self.max_frames
            #     c_frames = self.get_frames_from_segment(
            #         vid_id, start_, end_, num_frames=c_num_frames)
            #     if self.frames_noise != 0:
            #         clip_frames_noise = utils.truncated_normal_fill(
            #             c_frames.shape, std=self.frames_noise)
            #         c_frames += clip_frames_noise
            #     clip_frames_list.append(c_frames)
            #     clip_frames_len_list.append(c_frames.shape[0])
        else:
            duration = vid_dict['duration']
            vid_len = self.vid_data[vid_id].shape[0]
            for i in range(clip_num):
                start_ = int(vid_dict['timestamps'][i][0]/duration*vid_len)
                end_ = int(vid_dict['timestamps'][i][1]/duration*vid_len)
                c_num_frames = end_-start_
                if c_num_frames == 0:
                    c_num_frames = 1
                elif c_num_frames > self.max_frames:
                    c_num_frames = self.max_frames
                c_frames = self.get_frames_from_segment(
                    vid_id, start_, end_, num_frames=c_num_frames)
                if self.frames_noise != 0:
                    clip_frames_noise = utils.truncated_normal_fill(
                        c_frames.shape, std=self.frames_noise)
                    c_frames += clip_frames_noise
                clip_frames_list.append(c_frames)
                clip_frames_len_list.append(c_frames.shape[0])

        # load text
        seg_narrations = []
        for seg in vid_dict["sentences"]:
            seg_narrations.append(seg)
        list_of_list_of_words = self.preproc_par_fn(seg_narrations)

        # load precomputed text features
        par_cap_vectors, sent_cap_len_list = self.text_data[vid_id]
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
            "vid_id": vid_id, #video name
            "data_words": list_of_list_of_words, #sentences word
            "vid_frames": vid_frames, #video features
            "vid_frames_len": vid_frames_len, #video features len
            "par_cap_vectors": par_cap_vectors, #paragraph features
            "par_cap_len": par_cap_len, #paragraph features len
            "clip_num": clip_num,
            "sent_num": sent_num,
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


def create_datasets(
        dataset_path: str, cfg: EasyDict, preload_vid_feat: bool,
        preload_text_feat: bool):
    train_set = VideoDatasetFeatures(
        cfg.dataset.name, dataset_path, cfg.dataset.features,
        cfg.dataset.train_split, cfg.dataset.max_frames, True,
        preload_vid_feat, preload_text_feat, cfg.dataset.frames_noise)
    val_set = VideoDatasetFeatures(
        cfg.dataset.name, dataset_path, cfg.dataset.features,
        cfg.dataset.val_split, cfg.dataset.max_frames, False, preload_vid_feat,
        preload_text_feat, 0)
    return train_set, val_set


def create_loaders(
        train_set: VideoDatasetFeatures, val_set: VideoDatasetFeatures,
        batch_size: int, num_workers: int):
    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=train_set.collate_fn,
        pin_memory=True)
    val_loader = data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=val_set.collate_fn,
        pin_memory=True)
    return train_loader, val_loader

class ProposalDatasetFeatures(data.Dataset):
    def __init__(
            self, dataset_name, dataset_path, dataset_features,
            split, max_frames, is_train,
            preload_vid_feat, preload_text_feat,
            frames_noise):
        self.frames_noise = frames_noise
        self.split = split
        self.max_frames = max_frames
        self.is_train = is_train
        json_file = dataset_path / f"{split}.json"
        proposal_file = dataset_path / f"{split}_proposal.json"
        self.vids_dict = json.load(open(json_file, "r"))
        self.proposals_dict = json.load(open(proposal_file, "r"))['results']
        self.ids = list(self.vids_dict.keys())
        print(f"init dataset {dataset_name} split {split} length {len(self)} ")
        self.text_data = BertTextFeatureLoader(
            dataset_path, self.ids, preload_text_feat)
        self.preproc_par_fn = preprocess_bert_paragraph
        self.vid_data = ActivityNetVideoFeatureLoader(
            dataset_path, self.ids, preload_vid_feat)

    def get_frames_from_video(self, vid_id, indices=None, num_frames=None):
        vid_dict = self.vids_dict[vid_id]
        vid_len = self.vid_data[vid_id].shape[0]
        if num_frames is not None:
            indices = utils.compute_indices(
                vid_len, num_frames, self.is_train)
        frames = self.vid_data[vid_id][indices]
        return frames

    def get_frames_from_segment(
            self, vid_id, seg_num, clip_num, num_frames):
        start_frame = seg_num
        seg_len = int(clip_num - seg_num) # end-start
        indices = utils.compute_indices(seg_len, num_frames, self.is_train)
        indices += start_frame
        frames = self.get_frames_from_video(vid_id, indices)
        return frames

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        vid_id = self.ids[index]
        vid_dict = self.vids_dict[vid_id]
        proposal_dict = self.proposals_dict[vid_id]
        sent_num = len(vid_dict["sentences"])
        clip_num = len(proposal_dict)

        # load video frames
        vid_frames_len = self.vid_data[vid_id].shape[0]
        if vid_frames_len > self.max_frames:
            vid_frames_len = self.max_frames
        vid_frames = self.get_frames_from_video(
            vid_id, num_frames=vid_frames_len)
        vid_frames_len = int(vid_frames.shape[0])
        if self.frames_noise != 0:
            vid_frames_noise = utils.truncated_normal_fill(
                vid_frames.shape, std=self.frames_noise)
            vid_frames += vid_frames_noise

        # load segment frames
        clip_frames_list = []
        clip_frames_len_list = []
        duration = vid_dict['duration']
        vid_len = self.vid_data[vid_id].shape[0]
        for i in range(clip_num):
            start_ = int(proposal_dict[i]['segment'][0]/duration*vid_len)
            end_ = int(proposal_dict[i]['segment'][1]/duration*vid_len)
            c_num_frames = end_-start_
            if c_num_frames == 0:
                c_num_frames = 1
            elif c_num_frames > self.max_frames:
                c_num_frames = self.max_frames
            c_frames = self.get_frames_from_segment(
                vid_id, start_, end_, num_frames=c_num_frames)
            if self.frames_noise != 0:
                clip_frames_noise = utils.truncated_normal_fill(
                    c_frames.shape, std=self.frames_noise)
                c_frames += clip_frames_noise
            clip_frames_list.append(c_frames)
            clip_frames_len_list.append(c_frames.shape[0])

        # load text
        seg_narrations = []
        for seg in vid_dict["sentences"]:
            seg_narrations.append(seg)
        list_of_list_of_words = self.preproc_par_fn(seg_narrations)

        # load precomputed text features
        par_cap_vectors, sent_cap_len_list = self.text_data[vid_id]
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
            "vid_id": vid_id, #video name
            "data_words": list_of_list_of_words, #sentences word
            "vid_frames": vid_frames, #video features
            "vid_frames_len": vid_frames_len, #video features len
            "par_cap_vectors": par_cap_vectors, #paragraph features
            "par_cap_len": par_cap_len, #paragraph features len
            "clip_num": clip_num,
            "sent_num": sent_num,
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
    dataset_path = Path('../data')
    cfg = utils.load_config('./config/anet_coot.yaml')
    train_set, val_set = create_datasets(dataset_path, cfg, True, True)
    train_loader, val_loader = create_loaders(train_set, val_set, 1, 2)
    for batch_data in train_loader:
        for dt in batch_data:
            print(batch_data[dt])
        break