import argparse
import numpy as np
import glob
import os

from sklearn import metrics

def compute_eer(y, y_score):

	pred = [0 if x=='spoof' else 1 for x in y]

	fpr, tpr, thresholds = metrics.roc_curve(pred, y_score, pos_label=1)
	fnr = 1 - tpr
	eer_threshold = thresholds[np.nanargmin(np.abs(fnr-fpr))]
	eer = fpr[np.nanargmin(np.abs(fnr-fpr))]

	return eer

def read_scores(path):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	label_list, score_list = [], []

	for line in utt_labels:
		_, _, label, score = line.split(' ')
		label_list.append(label)
		score_list.append(float(score.strip('\n')))

	return score_list, label_list

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute EER given score files')
	parser.add_argument('--scores-path', type=str, default='./scores/', metavar='Path', help='Path to trials file')
	args = parser.parse_args()

	files_list = glob.glob(args.scores_path+'*.txt')

	print('considered files:')
	print(files_list)
	print(' ')

	for file_ in files_list:

		score_list, label_list = read_scores(file_)
		print('EER for file {}: {:0.4f}%'.format(file_.split('/')[-1], 100.*compute_eer(label_list, score_list)))

	print(' ')
	print('All done!!')
