import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path

import h5py
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from text_embedding import preprocess_bert_paragraph
torch.cuda.set_device(5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=list, default=['train', 'val_1'],
                        help="dataset name")
    parser.add_argument("--dataroot", type=str, default="../data",
                        help="change default path to dataset")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--bert_cache_path", type=str,
                        default="../bert_pytorch/", help="batch size")
    args = parser.parse_args()
    # load pretrained bert model
    print("load bert model...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_cache_path)
    model = AutoModel.from_pretrained(args.bert_cache_path, output_hidden_states=True)
    if args.cuda:
        model = model.cuda()
    dataset_path = Path(args.dataroot)

    # setup paths
    lengths_file = dataset_path / "text_lens.json"
    data_file = dataset_path / "text_features.hdf5"
    data_h5 = h5py.File(data_file, "w")
    lengths = {}
    for split in args.dataset_name:
        gt_path = Path(args.dataroot) / f"{split}.json"

        # load metadata
        vids_dict = json.load(open(gt_path, 'r'))
        layer_list_int = [-2, -1]

        # loop videos and encode features
        for vid_id, meta in tqdm(vids_dict.items(), desc="compute text features"):
            # collect narrations and preprocess them for bert
            sentences = [seg for seg in meta["sentences"]]
            paragraph = preprocess_bert_paragraph(sentences)
            # tokenize
            sent_tokens = []
            total_len = 0
            for list_of_words in paragraph:
                sentence = " ".join(list_of_words)
                sentence_int_tokens = tokenizer.encode(
                    sentence, add_special_tokens=False)
                sent_tokens.append(sentence_int_tokens)
                total_len += len(sentence_int_tokens)

            if total_len > 512:
                # cut too long tokens if needed, retaining some parts for all
                # sentences and all the SEP tokens
                s_caps_lens_old = [len(sentence) for sentence in sent_tokens]
                s_cap_lens = deepcopy(s_caps_lens_old)
                min_cut = 5
                for sent in range(len(paragraph) - 1, -1, -1):
                    overshoot = sum(s_cap_lens) - 512
                    if overshoot == 0:
                        break
                    new_len = max(min_cut, len(sent_tokens[sent]) - overshoot)
                    s_cap_lens[sent] = new_len
                sent_tokens_new = []
                for i, (old_len, new_len) in enumerate(zip(
                        s_caps_lens_old, s_cap_lens)):
                    if old_len != new_len:
                        sent_tokens_new.append(
                            sent_tokens[i][:new_len - 1] + [102])
                    else:
                        sent_tokens_new.append(sent_tokens[i])
                sent_tokens = sent_tokens_new

            # encode with bert
            sent_lens = [len(sentence) for sentence in sent_tokens]
            flat_paragraph = list(itertools.chain.from_iterable(sent_tokens))
            input_tensor = torch.tensor(flat_paragraph).long().unsqueeze(0)
            if args.cuda:
                input_tensor = input_tensor.cuda()
            _, _, layer_output = model(input_tensor)
            features = []
            for layer_num in layer_list_int:
                layer_features = layer_output[layer_num].squeeze(0)
                features.append(layer_features)
            features = torch.cat(features, -1)
            assert features.shape[0] == sum(sent_lens), ""

            # write features
            text_dict = data_h5.create_group(vid_id)
            text_dict['features'] = features.detach().cpu()
            lengths[vid_id] = sent_lens
    data_h5.close()

    # write lengths
    json.dump(lengths, lengths_file.open("wt", encoding="utf8"))


if __name__ == '__main__':
    main()
