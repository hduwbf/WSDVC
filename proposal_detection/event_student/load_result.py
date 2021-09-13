import json
import h5py
import torch
from tqdm import tqdm
video_features = h5py.File('../data/event/AnetFeatures.hdf5', 'r')
proposal_features = h5py.File('../data/event/ProposalFeatures.hdf5', 'w')
with open('./output/student_proposal.json', 'r') as file:
	proposal_file = json.load(file)['results']
	file.close()
with open('../data/event/val_1.json', 'r') as file:
	gt_file = json.load(file)
	file.close()
video_list = list(proposal_file.keys())
for v_id in tqdm(video_list):
	v_features = torch.tensor(video_features[v_id]['c3d_features']).cuda()
	duration = gt_file[v_id]['duration']
	proposal_feature = torch.zeros(100,100,500).cuda()
	proposal_mask = torch.zeros(100,100).cuda()
	for i in range(100):
		start = int(proposal_file[v_id][i]['segment'][0]/duration*100)
		end = int(proposal_file[v_id][i]['segment'][1]/duration*100)
		proposal_feature[i][0:end-start] = v_features[start, end]
		proposal_mask[i][0:end-start] = 1
	proposal_dict = proposal_features.create_group(v_id)
	proposal_dict['features'] = proposal_feature.cpu().detach()
	proposal_dict['mask'] = proposal_mask.cpu().detach()
	break
proposal_features.close()
proposal_features = h5py.File('../data/event/ProposalFeatures.hdf5', 'r')
print(proposal_features.keys())
