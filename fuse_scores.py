import argparse
import numpy as np
import glob
import os

from sklearn import metrics

def calibrate(scores):

	max_ = np.max(scores)
	min_ = np.min(scores)

	return (scores-min_)/(max_-min_)

def compute_MAD(scores_set, x_median):
	return np.median(np.abs(scores_set - x_median))

def is_outlier(x, x_median, MAD):
	M = np.abs(.6745*(x - x_median)/MAD)
	if M>3.5:
		return True
	else:
		return False

def get_non_outliers(scores_set):
	non_outliers = []
	median = np.median(scores_set)
	MAD = compute_MAD(scores_set, median)
	for score in scores_set:
		if not is_outlier(score, median, MAD):
			non_outliers.append(score)

	return non_outliers

def compute_eer(y, y_score):

	pred = [0 if x=='spoof' else 1 for x in y]

	fpr, tpr, thresholds = metrics.roc_curve(pred, y_score, pos_label=1)
	fnr = 1 - tpr

	t = np.nanargmin(np.abs(fnr-fpr))
	eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
	eer = (eer_low+eer_high)*0.5

	return eer

def read_trials(path, eval_=False):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	if eval_:

		utt_list = []

		for line in utt_labels:
			utt = line.strip('\n')
			utt_list.append(utt)

		return utt_list

	else:

		utt_list, attack_type_list, label_list = [], [], []

		for line in utt_labels:
			_, utt, _, attack_type, label = line.split(' ')
			utt_list.append(utt)
			attack_type_list.append(attack_type)
			label_list.append(label.strip('\n'))

		return utt_list, attack_type_list, label_list

def read_file(path, eval_=False):

	print('Reading data from file: {}'.format(path))

	with open(path, 'r') as file:
		utt_labels = file.readlines()

	utt_list, score_list = [], []

	if eval_:

		for line in utt_labels:
			utt, score = line.split(' ')
			utt_list.append(utt)
			score_list.append(float(score.strip('\n')))

		return utt_list, score_list

	else:

		for line in utt_labels:
			utt, _, _, score = line.split(' ')
			utt_list.append(utt)
			score_list.append(float(score.strip('\n')))

		return utt_list, score_list

def read_scores(path, calibrate_=False, eval_=False):

	data = {}

	if eval_:
		files_list = glob.glob(args.scores_path+'eval_*.txt')
	else:
		files_list = glob.glob(args.scores_path+'dev_*.txt')

	for file_ in files_list:

		utterances, scores = read_file(file_, eval_)

		if calibrate_:
			scores=calibrate(scores)

		for i in range(len(utterances)):
			if utterances[i] in data:
				data[utterances[i]].append(scores[i])
			else:
				data[utterances[i]] = [scores[i]]

	return data

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute EER given score files')
	parser.add_argument('--scores-path', type=str, default='./scores/', metavar='Path', help='Path to trials file')
	parser.add_argument('--out-path', type=str, default='./out.txt', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
	parser.add_argument('--eval', action='store_true', default=False, help='Enables eval trials reading')
	parser.add_argument('--no-output-file', action='store_true', default=False, help='Disables writing scores into out file')
	parser.add_argument('--no-eer', action='store_true', default=False, help='Disables computation of EER')
	parser.add_argument('--remove-outliers', action='store_true', default=False, help='Enables outlier removal')
	parser.add_argument('--calibrate', action='store_true', default=False, help='Enables calibrating scores for ensuring all are within the same range')
	args = parser.parse_args()

	if os.path.isfile(args.out_path):
		os.remove(args.out_path)
		print(args.out_path + ' Removed')

	if args.eval:
		test_utts = read_trials(args.trials_path, eval_=args.eval)
	else:
		test_utts, attack_type_list, label_list = read_trials(args.trials_path, eval_=args.eval)

	data = read_scores(args.scores_path, calibrate_=args.calibrate, eval_=args.eval)

	score_list = []

	for i, utt in enumerate(test_utts):

		print('Computing score for utterance '+ utt)

		scores = data[utt]

		if args.remove_outliers:
			score = np.mean(get_non_outliers(scores))
		else:
			score = np.mean(scores)

		score_list.append(score)

		print('Score: {}'.format(score_list[-1]))

	if not args.no_output_file:

		print('Storing scores in output file:')
		print(args.out_path)

		with open(args.out_path, 'w') as f:
			if args.eval:
				for i, utt in enumerate(test_utts):
					f.write("%s" % ' '.join([utt, str(score_list[i])+'\n']))
			else:
				for i, utt in enumerate(test_utts):
					f.write("%s" % ' '.join([utt, attack_type_list[i], label_list[i], str(score_list[i])+'\n']))

	if not args.no_eer and not args.eval:
		print('EER: {}'.format(compute_eer(label_list, score_list)))

	print('All done!!')
