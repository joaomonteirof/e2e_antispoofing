import argparse
import numpy as np
import glob
import torch
import torch.nn.functional as F
import os
from kaldi_io import read_mat_scp
import model as model_
import scipy.io as sio
from sklearn import preprocessing
from utils import compute_eer_labels, read_labels, get_utt2score

def prep_feats(data_):

	#data_ = ( data_ - data_.mean(0) ) / data_.std(0)

	features = data_.T

	if features.shape[1]<50:
		mul = int(np.ceil(50/features.shape[1]))
		features = np.tile(features, (1, mul))
		features = features[:, :50]

	return torch.from_numpy(features[np.newaxis, np.newaxis, :, :]).float()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute scores')
	parser.add_argument('--scores-path', type=str, default=None, metavar='Path', help='Path to scores file')
	parser.add_argument('--trials-path', type=str, default=None, metavar='Path', help='Path to trials file')
	parser.add_argument('--out-path', type=str, default='./out.txt', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--no-output-file', action='store_true', default=False, help='Disables writing scores into out file')
	args = parser.parse_args()

	if os.path.isfile(args.out_path):
		os.remove(args.out_path)
		print(args.out_path + ' Removed')

	test_utts, label_list = read_labels(args.trials_path)
	utt2score = get_utt2score(args.scores_path)
	lb = preprocessing.LabelBinarizer()
	y = torch.Tensor(lb.fit_transform(label_list)).squeeze(-1)

	print('Start of scores computation')

	score_list = []
	pred_list = []
	skipped_utterances = 0

	with torch.no_grad():

		for i, utt in enumerate(test_utts):

			try:
				score = utt2score[utt]
				pred = 1.-score
			except KeyError:
				print('\nSkipping utterance {}. Not found within the data\n'.format(utt))
				skipped_utterances+=1
				continue

			score_list.append(score)
			pred_list.append(pred)

	if not args.no_output_file:

		print('Storing scores in output file:')
		print(args.out_path)

		with open(args.out_path, 'w') as f:
			for i, utt in enumerate(test_utts):
				f.write("%s" % ' '.join([utt, str(score_list[i])+'\n']))

	print('EER: {}'.format(compute_eer_labels(label_list, score_list)))
	print('BCE: {}'.format(torch.nn.BCELoss()(torch.Tensor(pred_list), y)))

	print('All done!!')
	print('\nTotal skipped trials: {}'.format(skipped_utterances))
