# BAQM Thesis Tip Mining
*Erik van der Heide. Erasmus University Rotterdam 2021-2022.*

This repository contains all files used for my master thesis "TipBERT: Extracting Tips from Customer Reviews in E-commerce Using BERT".

The raw and preprocessed data are given in the folder 'Data'. In the folder 'Code', the code is split up into two: Data and Models. The Data files contain all code concerning data preparation, exploration and final evaluation. These are mainly .py files written in Anaconda's Spyder. The Models files contain all code related to the modeling part. These are mainly .ipynb files written in Google's Colaboratory.

## Data: Raw Data
* .

## Data: Preprocessed Data
* 

## Code Data
* ```data_training.py```. This file contains data preprocessing and exploration of the training data.
* ```data_full.py```. This file contains data preprocessing of full data and constructs evaluation data.
* ```data_meta.py```. This file is used to get the name of the products in the evaluation data.
* ```data_results.py```. This file is used to do calculations with the result data of the BERT model.

## Code Models
* ```BERT_Section5_1.py```. This file plots the sequence length of the training data.
* ```BERT_Section5_2.ipynb```. This file saves the [CLS] token and uses it as input for LR, XGB or ANN classifier.
* ```BERT_Section5_3.ipynb```. This file performs fine-tuning of BERT models for unbalanced and balanced data; weighted and unweighted cross-entropy; category specific and pooled data.
* ```BERT_Section5_4.ipynb```. This file aims to find the final best TipBERT model.
* ```BERT_Section5_5.ipynb```. This file performs TipBERT on new products.
* 
