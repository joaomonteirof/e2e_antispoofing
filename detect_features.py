import argparse
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from kaldi_io import read_vec_flt_scp
import glob
import os

def compute_eer(y, y_score):

	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr
	eer_threshold = thresholds[np.nanargmin(np.abs(fnr-fpr))]
	eer = fpr[np.nanargmin(np.abs(fnr-fpr))]

	return eer

def get_data_old(path1, path2):

	files_list_1 = glob.glob(path1 + 'ivector.*.ark')

	if not len(files_list_1)>0:
		raise ValueError('Nothing found at {}'.format(path1))

	features_1 = {}

	for file_ in files_list_1:
		features_1.update( { k:m for k,m in read_vec_flt_ark(file_) } )

	if path2 is not None:

		files_list_2 = glob.glob(path2 + 'ivector.*.ark')

		features_2 = {}

		for file_ in files_list_2:
			features_2.update( { k:m for k,m in read_vec_flt_ark(file_) } )

	x, y = [], []

	for k, m in features_1.items():
		utt_type = k.split('-')[-1]
		y.append(0 if utt_type=='spoof' else 1)

		if path2 is not None and k in features_2:
			x.append(np.concatenate([m, features_2[k]], 0))
		else:
			x.append(m)

	return np.asarray(x), np.asarray(y)

def get_data(path1, path2):

	file_ = path1 + 'ivector.scp'

	features_1 = { k:m for k,m in read_vec_flt_scp(file_) }

	if path2 is not None:

		file_ = path2 + 'ivector.scp'

		features_2 = { k:m for k,m in read_vec_flt_scp(file_) }

	x, y = [], []

	for k, m in features_1.items():
		utt_type = k.split('-')[-1]
		y.append(0 if utt_type=='spoof' else 1)

		if path2 is not None and k in features_2:
			x.append(np.concatenate([m, features_2[k]], 0))
		else:
			x.append(m)

	return np.asarray(x), np.asarray(y)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute embeddings')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Path to set of ark files containing features with naming following: ivectors.*.ark')
	parser.add_argument('--path2-to-data', type=str, default=None, metavar='Path', help='extra set of features for bi-model setting')
	parser.add_argument('--out-path', type=str, default='./trained_model.p', metavar='Path', help='Path to output trained model')
	parser.add_argument('--eval-dev', action='store_true', default=False, help='Enables evaluation on dev data')
	parser.add_argument('--no-out', action='store_true', default=False, help='Disables saving of best model')
	args = parser.parse_args()

	###############################################################################
	# Read data

	print('Loading train data')
	X, Y = get_data(path1=args.path_to_data+'ivectors_train/', path2=args.path2_to_data+'ivectors_train/' if args.path2_to_data is not None else None)
	print(X.shape, Y.shape)

	###############################################################################
	# Set up and run grid search

	print('Training model')
	forest = RandomForestClassifier()
	parameters = {'criterion':['gini', 'entropy'], 'n_estimators':[50, 100, 200, 400, 600]}
	clf = GridSearchCV(forest, parameters, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)

	clf.fit(X, Y)

	print('Training done!')

	###############################################################################
	# Printing results

	print('Random Forest')
	print('Best AUC: {}'.format(clf.best_score_) )
	print('Parameters yielding best AUC: {}'.format(clf.best_params_) )

	print('All results:')
	print(clf.cv_results_)

	###############################################################################
	# Saving final model

	if not args.no_out:
		print('Saving model')
		if os.path.isfile(args.out_path):
			os.remove(args.out_path)
		pickle.dump(clf.best_estimator_, open(args.out_path, 'wb'))

	###############################################################################
	# Evaluation on final data if provided

	if args.eval_dev:
		print('Evaluating EER on development data')
		X, Y = get_data(path1=args.path_to_data+'ivectors_dev/', path2=args.path2_to_data+'ivectors_dev/' if args.path2_to_data is not None else None)
		eer = compute_eer(Y, clf.best_estimator_.predict_proba(X)[:,1])
		print('EER: {}'.format(eer))
