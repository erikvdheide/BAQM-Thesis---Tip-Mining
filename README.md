# MSc Thesis Business Analytics & Quantative Marketing 
### TipBERT: Extracting Tips from Customer Reviews in E-commerce Using BERT
*Erik van der Heide. Erasmus University Rotterdam 2021.*

This repository contains all files used for my master thesis "TipBERT: Extracting Tips from Customer Reviews in E-commerce Using BERT".

The raw and preprocessed data are given in the folder 'Data'. In the folder 'Code', the code is split up into two: Data and Models. The Data files contain all code concerning data preparation, exploration and final evaluation. These are .py files written in Anaconda's Spyder. The Models files contain all code related to the modeling part. These are mainly .ipynb files written in Google's Colaboratory.

## Data: Raw Data
Contains the labeled dataset as provided by Hirsch et al. (2021) (```generate_tips_data```) including a version with some manually corrected line-break mistakes (```generate_tips_data_fixed```). Also a readme from Hirsch et al. (2021) is provided. Meta data and full reviews can be found on http://jmcauley.ucsd.edu/data/amazon/links.html.

## Data: Preprocessed Data
Contains the preprocessed datasets that are actually used to train and evaluate BERT models on. Consists of fully processed data (```td_clean```), including several balanced datasets (```td_clean_subsetX```, X = 1, 2, 3, 4, 5), as well as category-specific data and evaluation data of new products where TipBERT is evaluated on.

## Code: Code Data
* ```data_training.py```. This file contains data preprocessing and exploration of the full labeled training data set (85171 sentences, later split in train, val, test).
* ```data_full.py```. This file contains data preprocessing of non-labeled review data sets as can be found online (http://jmcauley.ucsd.edu/data/amazon/links.html) and constructs evaluation data for the final TipBERT (Section 5.5).
* ```data_meta.py```. This file is used to get the name of the products in the evaluation data.
* ```data_results.py```. This file is used to do calculations with the result data (Section 5.5) of the TipBERT model.

## Code: Code Models
* ```BERT_Section5_1.py```. This file plots the sequence length of the training data.
* ```BERT_Section5_2.ipynb```. This file saves the [CLS] token and uses it as input for LR, XGB or ANN classifier.
* ```BERT_Section5_3.ipynb```. This file performs fine-tuning of BERT models for unbalanced and balanced data; weighted and unweighted cross-entropy; category specific and pooled data.
* ```BERT_Section5_4.ipynb```. This file contains experiments to find the final best TipBERT model, which we use to evaluate new data on.
* ```BERT_Section5_5.ipynb```. This file trains TipBERT of the full data set and evaluates it on new products.
