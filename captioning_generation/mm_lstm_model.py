import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.weight_norm import weight_norm
from mm_dataset import *
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import pickle
from collections import defaultdict

def get_word_embed():
    translator_pickle = './translator.pkl'
    translator = pickle.load(open(translator_pickle, 'rb'))
    translator['word_to_id'] = defaultdict(lambda: len(translator['id_to_word'])-1, translator['word_to_id'])
    row = 0
    glove_file = '../data/glove.6B.300d.txt'
    words_embed = {}
    with open(glove_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
            if row > 30000:
                break
            row += 1
    id2emb = {}
    for ix in range(len(translator['id_to_word'])):
        if translator['id_to_word'][ix] in words_embed:
            id2emb[ix] = words_embed[translator['id_to_word'][ix]]
        else:
            id2emb[ix] = np.random.randn(300)
    data = [id2emb[ix] for ix in range(len(translator['id_to_word']))]
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(data)).to('cuda')
    return embedding

class Attention(nn.Module):
    """
    Attention Network.
    """
    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.2):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.features_att = weight_norm(nn.Linear(features_dim, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, visual_features, decoder_hidden, visual_mask):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.features_att(visual_features)  # (batch_size, 36, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout((att1 + att2.unsqueeze(1)).tanh())).squeeze(2)  # (batch_size, 36)
        att.masked_fill_(visual_mask==False, -float(1e8))
        alpha = self.softmax(att)  # (batch_size, 36)
        attention_weighted_encoding = (visual_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)
        return attention_weighted_encoding

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, features_dim, dropout=0.2):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(features_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = get_word_embed()  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.top_down_attention = nn.LSTMCell(embed_dim + features_dim + decoder_dim, decoder_dim, bias=True) # top down attention LSTMCell
        self.language_model = nn.LSTMCell(features_dim + decoder_dim, decoder_dim, bias=True)  # language model LSTMCell
        self.fc = weight_norm(nn.Linear(decoder_dim, vocab_size))  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self,batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size,self.decoder_dim).cuda()  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size,self.decoder_dim).cuda()
        return h, c

    def train_forward(self, visual_features, sent, visual_len, visual_mask):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = visual_features.shape[0]
        vocab_size = self.vocab_size
        output_prob = []
        output_pred = []
        # Flatten image
        visual_features_mean = visual_features.sum(1)/(visual_len.float().unsqueeze(1))  # (batch_size, num_pixels, encoder_dim)
        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)

        # Create tensors to hold word predicion scores
        output_prob.append((torch.zeros(batch_size, 1, self.vocab_size)).cuda())
        output_pred.append((torch.zeros(batch_size, 1)).long().cuda())
        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model
        for t in range(sent.shape[1]-1):
            embeddings = self.embedding(sent[:, t])
            h1,c1 = self.top_down_attention(
                torch.cat([h2,visual_features_mean,embeddings], dim=1), (h1, c1))
            attention_weighted_encoding = self.attention(visual_features, h1, visual_mask)
            h2,c2 = self.language_model(
                torch.cat([attention_weighted_encoding,h1], dim=1), (h2, c2))
            prob = F.log_softmax(self.fc(self.dropout(h2)), dim=1)  # (batch_size_t, vocab_size)
            _, pred = prob.max(1)
            output_prob.append(prob.unsqueeze(1))
            output_pred.append(pred.unsqueeze(1))
        return torch.cat(output_prob, dim=1), torch.cat(output_pred, dim=1), None, None

    def eval_forward(self, visual_features, visual_len, visual_mask, max_cap_len):
        batch_size = visual_features.shape[0]
        output_pred = []
        output_pred.append((torch.zeros(batch_size, 1)).long().cuda())
        visual_features_mean = visual_features.sum(1)/(visual_len.float().unsqueeze(1))  # (batch_size, num_pixels, encoder_dim)
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)

        next_input_word = torch.zeros(batch_size).long().cuda()
        for i in range(max_cap_len):
            embeddings = self.embedding(next_input_word)
            h1,c1 = self.top_down_attention(
                torch.cat([h2,visual_features_mean,embeddings], dim=1), (h1, c1))
            attention_weighted_encoding = self.attention(visual_features, h1, visual_mask)
            h2,c2 = self.language_model(
                torch.cat([attention_weighted_encoding,h1], dim=1), (h2, c2))
            prob = F.log_softmax(self.fc(self.dropout(h2)), dim=1)
            _, next_input_word = prob.max(1)
            output_pred.append(next_input_word.unsqueeze(1))
        output_pred = torch.cat(output_pred, dim=1)
        return None, output_pred, None, None

    def beam_search_forward(self, visual_features, visual_len, visual_mask, max_cap_len, beam_size):
        visual_features_mean = visual_features.sum(1)/(visual_len.float().unsqueeze(1))
        batch_size = visual_features_mean.shape[0]
        out_pred_target_list = list()
        out_pred_parent_list = list()
        candidate_score_dict = dict()
        current_scores = torch.FloatTensor(batch_size, beam_size).zero_().cuda()
        out_pred_target_list.append((torch.zeros(batch_size, beam_size).long().cuda()) + 0)
        out_pred_parent_list.append((torch.zeros(batch_size, beam_size).long().cuda()) - 1)
        current_scores[:, 1:].fill_(-float('inf'))
        h, c = self.init_hidden_state(batch_size)
        h1 = h.unsqueeze(1).repeat(1,beam_size,1).view(batch_size*beam_size, h.shape[1])
        h2 = h.unsqueeze(1).repeat(1,beam_size,1).view(batch_size*beam_size, h.shape[1])
        c1 = c.unsqueeze(1).repeat(1,beam_size,1).view(batch_size*beam_size, c.shape[1])
        c2 = c.unsqueeze(1).repeat(1,beam_size,1).view(batch_size*beam_size, c.shape[1])
        visual_features = visual_features.unsqueeze(1).repeat(1,beam_size,1,1).view(-1, visual_features.shape[1], visual_features.shape[2])
        visual_features_mean = visual_features_mean.unsqueeze(1).repeat(1,beam_size,1).view(-1, visual_features_mean.shape[1])
        visual_mask = visual_mask.unsqueeze(1).repeat(1,beam_size,1).view(-1, visual_mask.shape[1])
        next_input_word = torch.zeros(batch_size*beam_size).long().cuda()
        for step in range(1, max_cap_len+1):
            embeddings = self.embedding(next_input_word)
            h1,c1 = self.top_down_attention(
                torch.cat([h2,visual_features_mean,embeddings],dim=1), (h1,c1))
            attention_weighted_encoding = self.attention(visual_features, h1, visual_mask)
            h2,c2 = self.language_model(
                torch.cat([attention_weighted_encoding,h1], dim=1), [h2,c2])
            output = F.log_softmax(self.fc(self.dropout(h2)), dim=1)
            output_scores = output.view(batch_size, beam_size, -1)
            output_scores = output_scores + current_scores.unsqueeze(2)
            current_scores, output_candidate = output_scores.view(batch_size, -1).topk(beam_size, dim=1)
            next_input_word = (output_candidate % self.vocab_size).view(-1)
            parents = (output_candidate / self.vocab_size).view(batch_size, beam_size)
            hidden_gather_idx = parents.view(batch_size, beam_size, 1).expand(batch_size, beam_size, h1.shape[1])
            h1 = h1.view(batch_size, beam_size, h1.shape[1]).gather(dim=1, index=hidden_gather_idx).view(batch_size*beam_size, h1.shape[1])
            h2 = h2.view(batch_size, beam_size, h2.shape[1]).gather(dim=1, index=hidden_gather_idx).view(batch_size*beam_size, h2.shape[1])
            c1 = c1.view(batch_size, beam_size, c1.shape[1]).gather(dim=1, index=hidden_gather_idx).view(batch_size*beam_size, c1.shape[1])
            c2 = c2.view(batch_size, beam_size, c2.shape[1]).gather(dim=1, index=hidden_gather_idx).view(batch_size*beam_size, c2.shape[1])
            out_pred_target_list.append(next_input_word.view(batch_size, beam_size))
            out_pred_parent_list.append(parents)
            end_mask = next_input_word.data.eq(1).view(batch_size, beam_size)
            if end_mask.nonzero().dim()>0:
                stored_scores = current_scores.clone()
                current_scores.masked_fill_(end_mask, -float('inf'))
                stored_scores.data.masked_fill_(end_mask==False, -float('inf'))
                candidate_score_dict[step] = stored_scores
        final_pred = list()
        seq_length = torch.LongTensor(batch_size).zero_().cuda() + 1
        max_score, current_idx = current_scores.max(1)
        current_idx = current_idx.unsqueeze(1)
        final_pred.append((torch.zeros(batch_size, 1).long().cuda()) + 1)
        for step in range(max_cap_len, 0, -1):
            if step in candidate_score_dict:
                max_score, true_idx = torch.cat([candidate_score_dict[step], max_score.unsqueeze(1)], dim=1).max(1)
                current_idx[true_idx != beam_size,0] = true_idx[true_idx != beam_size]
                seq_length[true_idx != beam_size] = 0
            final_pred.append(out_pred_target_list[step].gather(dim=1, index=current_idx))
            current_idx = out_pred_parent_list[step].gather(dim=1, index=current_idx)
            seq_length = seq_length + 1
        final_pred.append(out_pred_target_list[0].gather(dim=1, index=current_idx))
        seq_length = seq_length + 1
        final_pred = torch.cat(final_pred[::-1], dim=1)
        caption_mask = torch.LongTensor(batch_size, max_cap_len + 2).zero_().cuda()
        caption_mask_helper = torch.LongTensor(range(max_cap_len + 2)).unsqueeze(0).repeat(batch_size, 1).cuda()
        caption_mask[caption_mask_helper < seq_length.unsqueeze(1)] = 1
        return None, final_pred.detach(), seq_length.detach(), caption_mask.detach()

    def forward(self, visual_features, sent, visual_len, visual_mask, max_cap_len, beam_size):
        if self.training:
            return self.train_forward(visual_features, sent, visual_len, visual_mask)
        else:
            return self.beam_search_forward(visual_features, visual_len, visual_mask, max_cap_len, beam_size)
            # return self.eval_forward(visual_features, visual_len, visual_mask, max_cap_len)
            return self.train_forward(visual_features, sent, visual_len, visual_mask)

class CaptionGenerator(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, features_dim, dropout, max_cap_length):
        super(CaptionGenerator, self).__init__()
        self.generator = DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size, features_dim, dropout)
        self.max_cap_length = max_cap_length
    def forward(self, video_feature, video_length, video_mask, temp_seg, seg_gather_idx, sent=None, beam_size=3):
        video_feature = video_feature.index_select(dim=0, index=seg_gather_idx)
        batch_size = video_feature.shape[0]
        video_seq_len, _ = video_length.index_select(dim=0, index=seg_gather_idx).chunk(2, dim=1)
        video_seq_len = video_seq_len.contiguous()
        video_mask = video_mask.index_select(dim=0, index=seg_gather_idx)
        start_index, end_index = self.index_extractor(temp_seg)
        clip_len = end_index-start_index+1
        max_clip_len = max(clip_len).item()
        clip_mask = torch.zeros(batch_size, max_clip_len).cuda()
        clip_feature = torch.zeros(batch_size, max_clip_len, video_feature.shape[2]).cuda()
        for i in range(batch_size):
            clip_feature[i, :clip_len[i]] = video_feature[i, start_index[i]:end_index[i]+1]
            clip_mask[i, :clip_len[i]] = 1
            # if clip_len[i]<0:
            #     clip_len[i] = -clip_len[i]
            #     clip_feature[i, :clip_len[i]] = video_feature[i, end_index[i]:start_index[i]+1]
            # elif clip_len[i]==0:
            #     clip_len[i] = 1
            #     clip_feature[i, :clip_len[i]] = video_feature[i, start_index[i]:end_index[i]+1]
            # else:
            #     clip_feature[i, :clip_len[i]] = video_feature[i, start_index[i]:end_index[i]+1]
            # clip_mask[i, :clip_len[i]] = 1
        prob, pred, sent_len, sent_mask = self.generator(clip_feature, sent, clip_len, clip_mask, self.max_cap_length, beam_size=beam_size)
        return prob, pred, sent_len, sent_mask

    def index_extractor(self, temp_seg):
        s, e = temp_seg.chunk(2, dim=1)
        return (s.squeeze(1)).long(), (e.squeeze(1)).long()

    def build_loss(self, caption_prob, ref_caption, caption_mask):
        assert caption_prob.size(1) == ref_caption.size(1)
        prob = caption_prob.gather(dim=2, index=ref_caption.unsqueeze(2))
        return - (prob * caption_mask).sum() / prob.size(0)

if __name__ == '__main__':
    model = CaptionGenerator(512, 300, 512, 6000, 2048, 0.2, 20)
    model = model.cuda()
    # model = model.cuda()
    training_set = WSDataEval('../data/ws_train.json', '../data/MultiModalFeature.hdf5', './translator.pkl', 2)
    train_loader = data.DataLoader(training_set, batch_size=4, shuffle=False,
        num_workers=1, collate_fn=collate_fn, drop_last=True)
    pred_dict = {'version': 'V0',
                 'results':{},
                 'external_data':{
                    'used': True,
                     'details': 'provided MM feature'
                 }}
    for idx, batch_data in enumerate(train_loader):
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
        caption_prob, _,_,_ = model.forward(video_feat, video_len, video_mask, ts_seq, sent_gather_idx, sent_feat)
        loss = model.build_loss(caption_prob, sent_feat, sent_mask)
        print(loss)
        break