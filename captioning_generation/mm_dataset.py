import pickle
import numpy as np
import torch.utils.data as data
import json
from collections import defaultdict
import h5py
from itertools import chain
import torch

def collate_fn(batch):
    batch_size = len(batch)
    feature_size = batch[0][0].shape[1]
    feature_list, timestamps_list, caption_list, raw_timestamp, raw_duration, key = zip(*batch)
    max_video_length = max([x.shape[0] for x in feature_list])
    max_caption_length = max(chain(*[[len(caption) for caption in captions] for captions in caption_list]))
    total_caption_num = sum(chain([len(captions) for captions in caption_list]))
    video_tensor = torch.FloatTensor(batch_size, max_video_length, feature_size).zero_()
    video_length = torch.FloatTensor(batch_size, 2).zero_()
    video_mask = torch.FloatTensor(batch_size, max_video_length, 1).zero_()
    timestamps_tensor = torch.FloatTensor(total_caption_num, 2).zero_()
    caption_tensor = torch.LongTensor(total_caption_num, max_caption_length).zero_() + 1
    caption_length = torch.LongTensor(total_caption_num).zero_()
    caption_mask = torch.FloatTensor(total_caption_num, max_caption_length, 1).zero_()
    caption_gather_idx = torch.LongTensor(total_caption_num).zero_()
    total_caption_idx = 0
    for idx in range(batch_size):
        video_len = feature_list[idx].shape[0]
        #video
        video_tensor[idx, :video_len, :] = torch.Tensor(feature_list[idx])
        video_length[idx, 0] = float(video_len)
        video_length[idx, 1] = raw_duration[idx]
        video_mask[idx, :video_len, 0] = 1
        #timestamps
        proposal_length = len(timestamps_list[idx])
        timestamps_tensor[total_caption_idx:total_caption_idx+proposal_length, :] = \
            torch.Tensor(timestamps_list[idx])
        caption_gather_idx[total_caption_idx:total_caption_idx+proposal_length] = idx
        #caption
        for iidx, captioning in enumerate(caption_list[idx]):
            _caption_len = len(captioning)
            caption_length[total_caption_idx+iidx] = _caption_len
            caption_tensor[total_caption_idx+iidx, :_caption_len] = torch.Tensor(captioning)
            caption_mask[total_caption_idx+iidx, :_caption_len, 0] = 1
        total_caption_idx += proposal_length
    raw_timestamp = torch.FloatTensor(list(chain(*raw_timestamp)))
    # print((video_length[:,0] / video_length[:,1]).mean())
    return (video_tensor, video_length, video_mask,
            caption_tensor, caption_length, caption_mask, caption_gather_idx,
            raw_timestamp, timestamps_tensor, key)

class WSDataset(data.Dataset):
    def __init__(self, caption_file, feature_file, translator_pickle, sample_rate):
        super(WSDataset, self).__init__()
        self.caption_file = json.load(open(caption_file, 'r'))
        self.feature_file = h5py.File(feature_file, 'r')
        self.keys = list(set(self.caption_file.keys()).intersection(self.feature_file.keys()))
        self.translator = pickle.load(open(translator_pickle, 'rb'))
        self.translator['word_to_id'] = defaultdict(lambda: len(self.translator['id_to_word'])-1,
            self.translator['word_to_id'])
        self.sample_rate = sample_rate
    def __len__(self):
        return len(self.keys)
    def sent2id(self, sentence):
        sentence_split = sentence.replace('.', ' . ').replace(',', ' , ').lower().split()
        sentence_split = ['<bos>'] + sentence_split + ['<eos>']
        res = [self.translator['word_to_id'][word] for word in sentence_split]
        return res
    def id2sent(self, sent_ids):
        assert sent_ids[0] == self.translator['word_to_id']['<bos>']
        sent_ids = sent_ids[1:]
        for i in range(len(sent_ids)):
            if sent_ids[i] == self.translator['word_to_id']['<eos>']:
                sent_ids = sent_ids[:i]
                break
        return ' '.join([self.translator['id_to_word'][idx] for idx in sent_ids])
    def process_visual_segment(self, duration, timestamps_list, feature_length):
        res = np.zeros([len(timestamps_list), 2])
        for idx, (start, end) in enumerate(timestamps_list):
            if start>end:
                holder = end
                end = start
                start = holder
            elif start == end:
                end = end + 1
            start_, end_ = int(start/duration*feature_length), min(feature_length-1, int(end/duration*feature_length))
            res[idx] = [start_, end_]
        return res
    def __getitem__(self, idx):
        raise NotImplementedError()

class WSDataFull(WSDataset):
    def __init__(self, caption_file, feature_file, translator_pickle, sample_rate):
        super(WSDataFull, self).__init__(caption_file,
            feature_file, translator_pickle, sample_rate)
    def __getitem__(self, idx):
        key = str(self.keys[idx])
        feature_obj = self.feature_file[key]['features']
        feature_obj = feature_obj[::self.sample_rate, :]
        captioning = self.caption_file[key]['sentences']
        full_cap = ''
        for sent in captioning:
            full_cap += sent
        captioning = [self.sent2id(full_cap)]
        duration = self.caption_file[key]['duration']
        timestamps = [[0, duration]]
        processed_timestamps = self.process_visual_segment(duration, timestamps, feature_obj.shape[0])
        return feature_obj, processed_timestamps, captioning, timestamps, duration, key

class WSDataEval(WSDataset):
    def __init__(self, caption_file, feature_file, translator_pickle, sample_rate):
        super(WSDataEval, self).__init__(caption_file,
            feature_file, translator_pickle, sample_rate)
    def __getitem__(self, idx):
        key = str(self.keys[idx])
        feature_obj = self.feature_file[key]['features']
        feature_obj = feature_obj[::self.sample_rate, :]
        captioning = self.caption_file[key]['sentences']
        captioning = [self.sent2id(sent) for sent in captioning]
        timestamps = self.caption_file[key]['timestamps']
        duration = self.caption_file[key]['duration']
        processed_timestamps = self.process_visual_segment(duration, timestamps, feature_obj.shape[0])
        return feature_obj, processed_timestamps, captioning, timestamps, duration, key

class WSDataProposal(WSDataset):
    def __init__(self, caption_file, feature_file, translator_pickle, sample_rate):
        super(WSDataProposal, self).__init__(caption_file,
            feature_file, translator_pickle, sample_rate)
        self.proposal_file = json.load(open('../data/select_proposal.json', 'r'))['results']
    def __getitem__(self, idx):
        key = str(self.keys[idx])
        feature_obj = self.feature_file[key]['features']
        feature_obj = feature_obj[::self.sample_rate, :]
        captioning = [self.sent2id('None') for i in range(len(self.proposal_file[key]))]
        timestamps = [self.proposal_file[key][i]['segment'] for i in range(len(self.proposal_file[key]))]
        duration = self.caption_file[key]['duration']
        processed_timestamps = self.process_visual_segment(duration, timestamps, feature_obj.shape[0])
        return feature_obj, processed_timestamps, captioning, timestamps, duration, key

class WSDataTrain(WSDataset):
    def __init__(self, caption_file, feature_file, translator_pickle, sample_rate):
        super(WSDataTrain, self).__init__(caption_file,
            feature_file, translator_pickle, sample_rate)
    def __getitem__(self, idx):
        key = str(self.keys[idx])
        feature_obj = self.feature_file[key]['features']
        feature_obj = feature_obj[::self.sample_rate, :]
        captioning = self.caption_file[key]['sentences']
        if captioning == None:
            print(self.caption_file[key])
        idx = int(np.random.choice(range(len(captioning)), 1))
        captioning = [self.sent2id(captioning[idx])]#[[self.sent2id(sent) for sent in captioning][idx]]
        timestamps = [self.caption_file[key]['timestamps'][idx]]
        duration = self.caption_file[key]['duration']
        processed_timestamps = self.process_visual_segment(duration, timestamps, feature_obj.shape[0])
        return feature_obj, processed_timestamps, captioning, timestamps, duration, key

if __name__ == '__main__':
    dataset = WSDataTrain('../data/train.json',
                       '../data/MultiModalFeature.hdf5',
                       './translator.pkl', 2)
    data_loader = data.DataLoader(dataset, batch_size=16,
        shuffle=True, num_workers=1, collate_fn=collate_fn)
    for dt in data_loader:
        print(dt[2])
        for tensor in dt[:-1]:
            print(type(tensor), tensor.size())
        print(dt[-1][0])
        print('*'*80)
        break