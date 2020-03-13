import numpy as np
from sklearn import metrics

import torch

import os
import sys
import pickle
from time import sleep

def set_np_randomseed(worker_id):
	np.random.seed(np.random.get_state()[1][0]+worker_id)

def get_freer_gpu(trials=10):
	sleep(5)
	for j in range(trials):
		os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
		memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
		dev_ = torch.device('cuda:'+str(np.argmax(memory_available)))
		try:
			a = torch.rand(1).cuda(dev_)
			return dev_
		except:
			pass

	print('NO GPU AVAILABLE!!!')
	exit(1)

def set_device(trials=10):
	a = torch.rand(1)

	for i in range(torch.cuda.device_count()):
		for j in range(trials):

			torch.cuda.set_device(i)
			try:
				a = a.cuda()
				print('GPU {} selected.'.format(i))
				return
			except:
				pass

	print('NO GPU AVAILABLE!!!')
	exit(1)

def compute_eer_labels(labels, y_score):

	pred = [0 if x=='spoof' else 1 for x in labels]

	fpr, tpr, thresholds = metrics.roc_curve(pred, y_score, pos_label=1)
	fnr = 1 - tpr

	t = np.nanargmin(np.abs(fnr-fpr))
	eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
	eer = (eer_low+eer_high)*0.5

	return eer

def compute_eer(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr

	t = np.nanargmin(np.abs(fnr-fpr))
	eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
	eer = (eer_low+eer_high)*0.5

	return eer

def compute_metrics(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr
	t = np.nanargmin(np.abs(fnr-fpr))

	eer_threshold = thresholds[t]

	eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
	eer = (eer_low+eer_high)*0.5

	auc = metrics.auc(fpr, tpr)

	avg_precision = metrics.average_precision_score(y, y_score)

	pred = np.asarray([1 if score > eer_threshold else 0 for score in y_score])
	acc = metrics.accuracy_score(y ,pred)

	return eer, auc, avg_precision, acc, eer_threshold

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

def read_labels(path):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	utt_list, label_list = [], []

	for line in utt_labels:
		utt, label = line.split(' ')
		utt_list.append(utt)
		label_list.append(label.strip('\n'))

	return utt_list, attack_type_list, label_list

def change_keys(data_dict):

	keys_=list(data_dict.keys())

	for i in range(len(keys_)):
		k = keys_[i]
		new_k = k.split('-')[0]
		data_dict[new_k] = data_dict.pop(k)

	return data_dict

def get_utt2score(path):
	with open(path, 'r') as file:
		rows = file.readlines()

	utt2score_dict = {}

	for row in rows:
		utt_score = row.strip('\n').split(' ')
		utt2score_dict[utt_score[0]] = float(utt_score[1:])

	return utt2score_dict

def read_spk2utt(path):
	with open(path, 'r') as file:
		rows = file.readlines()

	spk2utt_dict = {}

	for row in rows:
		spk_utts = row.replace('\n','').split(' ')
		spk2utt_dict[spk_utts[0]] = spk_utts[1:]

	return spk2utt_dict

def read_scores(path):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	label_list, score_list = [], []

	for line in utt_labels:
		_, _, label, score = line.split(' ')
		label_list.append(label)
		score_list.append(float(score.strip('\n')))

	return score_list, label_list

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
