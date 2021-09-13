# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
# from caption_model import *
from mm_lstm_model import *
from mm_dataset import *
# model = CaptionGenerator(300, 4, 'GRU', 0.3, False, 0.1, 6000, 300, 500, True, 20)
model = CaptionGenerator(512, 300, 512, 6000, 2048, 0.2, 20)
model = model.cuda()
proposal_val_set = WSDataProposal('../data/val_1.json', '../data/MultiModalFeature.hdf5', './translator.pkl', 2)
gt_val_set = WSDataEval('../data/val_1.json', '../data/MultiModalFeature.hdf5', './translator.pkl', 2)
val_loader = data.DataLoader(proposal_val_set, batch_size=1, shuffle=False,
    num_workers=1, collate_fn=collate_fn, drop_last=False)
pred_dict = {'version': 'V0',
             'results':{},
             'external_data':{
                'used': True,
                 'details': 'provided MultiModal feature'
             }}
checkpoint = torch.load('./checkpoint/mm_ours.pth.tar')#, map_location={'cuda:13':'cuda:0'})
model.load_state_dict(checkpoint['state_dict'])
model.eval()
for idx, batch_data in enumerate(val_loader):
    with torch.no_grad():
        video_feat, video_len, video_mask, sent_feat, _, _, sent_gather_idx, ts_time, ts_seq, key = batch_data
        video_feat = video_feat.cuda()
        video_len = video_len.cuda()
        video_mask = video_mask.cuda()
        sent_feat = sent_feat.cuda()
        sent_gather_idx = sent_gather_idx.cuda()
        ts_seq = ts_seq.cuda()
        video_seq_len, video_time_len = video_len.index_select(dim=0, index=sent_gather_idx).chunk(2, dim=1)
        # _, caption_pred, _, _ = model.forward(video_feat, video_len, video_mask, ts_seq, sent_gather_idx)
        _, caption_pred, _, _ = model.forward(video_feat, video_len, video_mask, ts_seq, sent_gather_idx)
        _ts_time = (ts_seq/video_seq_len*video_time_len).cpu().data.numpy()
        caption_pred = caption_pred.cpu().data.numpy()
        for idx in range(len(sent_gather_idx)):
            video_key = key[sent_gather_idx.cpu().data[idx]]
            if video_key not in pred_dict['results']:
                pred_dict['results'][video_key] = list()
            pred_dict['results'][video_key].append({
                'sentence': val_loader.dataset.id2sent(caption_pred[idx]),
                'timestamp': _ts_time[idx].tolist()
            })
pred_dict = json.dumps(pred_dict)
with open('../data/mm_ours_proposal.json', 'w') as file:
    file.write(pred_dict)
    file.close()
print('predict is done')