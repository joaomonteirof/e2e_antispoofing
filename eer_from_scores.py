import argparse
import numpy as np
import glob
import os

from sklearn import metrics

def compute_eer(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr

	t = np.nanargmin(np.abs(fnr-fpr))
	eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
	eer = (eer_low+eer_high)*0.5

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
