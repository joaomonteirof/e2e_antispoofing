import argparse
import numpy as np
import glob
import os

from utils import compute_eer_labels, read_scores

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
		print('EER for file {}: {:0.4f}%'.format(file_.split('/')[-1], 100.*compute_eer_labels(label_list, score_list)))

	print(' ')
	print('All done!!')
