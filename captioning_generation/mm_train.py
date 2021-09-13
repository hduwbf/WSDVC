import torch
# import time
from mm_dataset import *
from mm_lstm_model import *
import argparse
import torch.utils.data as data
import time

def train(model, data_loader, epoch, optimizer):
    print('Train Phase...')
    model.train()
    _start_time = time.time()
    accumulate_loss = 0
    for idx, batch_data in enumerate(data_loader):
        batch_time = time.time()
        video_feat, video_len, video_mask, sent_feat, sent_len, sent_mask, sent_gather_idx, _, ts_seq, _ = batch_data
        video_feat = video_feat.cuda()
        video_len = video_len.cuda()
        video_mask = video_mask.cuda()
        sent_feat = sent_feat.cuda()
        sent_len = sent_len.cuda()
        sent_mask = sent_mask.cuda()
        sent_gather_idx = sent_gather_idx.cuda()
        ts_seq = ts_seq.cuda()
        # ts_seq[:, 0] = 0
        # ts_seq[:, 1] = video_feat.shape[1]-1
        video_seq_len, _ = video_len.index_select(dim=0, index=sent_gather_idx).chunk(2, dim=1)
        caption_prob, _, _, _ = model.forward(video_feat, video_len, video_mask, ts_seq, sent_gather_idx, sent_feat)
        loss = model.build_loss(caption_prob, sent_feat, sent_mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 0.25)
        optimizer.step()
        accumulate_loss += loss.item()
        if idx % 20 == 0 and idx != 0:
            print('train: epoch %03d, batch[%04d/%04d], time=%0.4fs, loss: %06.6f' %
                (epoch, idx, len(data_loader), time.time()-batch_time, loss.item()))
    print('epoch %03d, time:%0.4fs, avg loss: %06.6f' %
        (epoch, time.time()-_start_time, accumulate_loss/len(data_loader)))
    state = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, './checkpoint/mm_baseline.pth.tar')

def eval(model, data_loader, epoch, optimizer, best_loss):
    print('Validation Phase...')
    model.eval()
    _start_time = time.time()
    accumulate_loss = 0
    for idx, batch_data in enumerate(data_loader):
        with torch.no_grad():
            batch_time = time.time()
            video_feat, video_len, video_mask, sent_feat, sent_len, sent_mask, sent_gather_idx, _, ts_seq, _ = batch_data
            video_feat = video_feat.cuda()
            video_len = video_len.cuda()
            video_mask = video_mask.cuda()
            sent_feat = sent_feat.cuda()
            sent_len = sent_len.cuda()
            sent_mask = sent_mask.cuda()
            sent_gather_idx = sent_gather_idx.cuda()
            ts_seq = ts_seq.cuda()
            video_seq_len, _ = video_len.index_select(dim=0, index=sent_gather_idx).chunk(2, dim=1)
            caption_prob, _, _, _ = model.forward(video_feat, video_len, video_mask, ts_seq, sent_gather_idx, sent_feat)
            loss = model.build_loss(caption_prob, sent_feat, sent_mask)
            accumulate_loss += loss.item()
    print('epoch %03d, time:%0.4fs, avg loss: %06.6f' %
        (epoch, time.time()-_start_time, accumulate_loss/len(data_loader)))
    if accumulate_loss/len(data_loader)<best_loss:
        best_loss = accumulate_loss/len(data_loader)
        state = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, './checkpoint/mm_baseline_best.pth.tar')
    return best_loss

def main(params):
    model = CaptionGenerator(
        params['attention_dim'], params['embed_dim'], params['decoder_dim'], params['vocab_size'], params['features_dim'], params['dropout'], params['max_cap_length'])
    best_loss = 1e10
    model = model.cuda()
    # checkpoint = torch.load('./checkpoint/mm_pretrain_best.pth.tar')
    # model.load_state_dict(checkpoint['state_dict'])
    # model = torch.nn.DataParallel(model, device_ids=range(params['gpu_num'])).cuda()
    training_set = WSDataTrain(params['train_data'], params['feature_path'], params['translator_path'], params['video_sample_rate'])
    val_set = WSDataEval(params['val_data'], params['feature_path'], params['translator_path'], params['video_sample_rate'])
    ws_set = WSDataTrain(params['ws_data'], params['feature_path'], params['translator_path'], params['video_sample_rate'])
    train_loader = data.DataLoader(training_set, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], collate_fn=collate_fn, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=params['batch_size'], shuffle=False,
        num_workers=params['num_workers'], collate_fn=collate_fn, drop_last=False)
    ws_loader = data.DataLoader(ws_set, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], collate_fn=collate_fn, drop_last=True)
    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        # lr=params['lr'], weight_decay=params['weight_decay'], momentum=params['momentum'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=params['lr_step'], gamma=params["lr_decay_rate"])
    for epoch in range(params['training_epoch']):
        lr_scheduler.step()
        train(model, train_loader, epoch, optimizer)
        if (epoch+1) % params['test_interval'] == 0:
            best_loss = eval(model, val_loader, epoch, optimizer, best_loss)
    print('training finished successfully!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # datasets
    parser.add_argument('--train_data', type=str, default='../data/train.json')
    parser.add_argument('--val_data', type=str, default='../data/val_1.json')
    parser.add_argument('--ws_data', type=str, default='../data/ws_train_threshold.json')
    parser.add_argument('--feature_path', type=str, default='../data/MultiModalFeature.hdf5')
    parser.add_argument('--vocab_size', type=int, default=6000)
    parser.add_argument('--translator_path', type=str, default='./translator.pkl')
    parser.add_argument('--video_sample_rate', type=int, default=2)
    # model setting
    parser.add_argument('--max_cap_length', type=int, default=20)
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--features_dim', type=int, default=2048)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gpu_num', type=int, default=8)
    # training setting
    parser.add_argument('--optim_method', type=str, default='SGD')
    parser.add_argument('--training_epoch', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=1,
                        help='used in data loader(only 1 is supported because of bugs in h5py)')
    parser.add_argument('--test_interval', type=int, default=4)
    parser.add_argument('--lr_step', type=int, nargs='+', default=[32, 48])
    params = parser.parse_args()
    params = vars(params)

    main(params)