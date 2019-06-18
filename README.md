# End-to-end spoofing detection

End-to-end detection of spoofing attacks of various types (synthectic and replay) to Automatic speaker Verification systems

## Requirements

```
Python = 3.6
Pytorch >= 1.0.0
Scikit-learn >=0.19
tqdm
h5py
```

## Prepare data

Data preparation scripts are provided and features in [Kaldi](https://kaldi-asr.org/) format are exepected.

Pre-processed data will consist of separate hdf files for each class (clean/attack) such that features for each recording are stored as datasets of shape [1, nfeat, nframes].

Prepare Kaldi features with data_prep.py. Arguments:

```
--path-to-data        Path to feats.scp
--path-to-more-data   Path to second feats.scp
--out-path            Path to output hdf file
--out-name            Output hdf file name
--n-val-speakers      Number of speakers for valid data
```

Experiments are performed with several input features. Assumed input shapes for each model can be seen in test_arch.py (nframes can vary freely).

Models indicated by CC support a varying number of input features.

## Train a model

Train models with train.py. Arguments:

```
--model               {lstm,resnet,resnet_pca,lcnn_9,lcnn_29,lcnn_9_pca,lcnn_29_pca,lcnn_9_prodspec,lcnn_9_icqspec,lcnn_9_CC,lcnn_29_CC,resnet_34_CC}
--batch-size          input batch size for training (default: 64)
--epochs              number of epochs to train (default: 500)
--lr                  learning rate (default: 0.001)
--momentum alpha      Alpha (default: 0.9)
--l2                  Weight decay coefficient (default: 0.00001)
--checkpoint-epoch    epoch to load for checkpointing. If None, training starts from scratch
--checkpoint-path     Path for checkpointing
--pretrained-path     Path for pre trained model
--train-hdf-path      Path to hdf data
--valid-hdf-path      Path to hdf data
--workers WORKERS     number of data loading workers
--seed                random seed (default: 1)
--save-every          how many epochs to wait before logging training status. Default is 1
--n-frames            maximum number of frames per utterance (default: 1000)
--n-cycles            number of examples to complete 1 epoch
--valid-n-cycles      number of examples to complete 1 epoch
--n-classes           Number of classes for the mcc case (default: binary classification)
--ncoef               Number of cepstral coefs for the LA case (default: 90)
--init-coef           First cepstral coefs (default: 0)
--lists-path          Path to list files per attack
--no-cuda             Disables GPU use
```

## Scoring test recordings

Score models with score.py. Arguments:

```
--path-to-data        Path to input data
--trials-path         Path to trials file
--cp-path Path        Path for file containing model
--out-path            Path to output hdf file
--model               {lstm,resnet,resnet_pca,lcnn_9,lcnn_29,lcnn_9_pca,lcnn_29_pca,lcnn_9_prodspec,lcnn_9_icqspec,lcnn_9_CC,lcnn_29_CC,resnet_34_CC}
--no-cuda             Disables GPU use
--no-output-file      Disables writing scores into out file
--no-eer              Disables computation of EER
--eval                Enables eval trials reading
--tandem              Scoring with tandem features
--ncoef               Number of cepstral coefs (default: 90)
--init-coef           First cepstral coefs (default: 0)
```

End-2-end EER will printed in the screen and scores for each trial output in a file.
Input data is expected in kaldi format (feats.scp).
