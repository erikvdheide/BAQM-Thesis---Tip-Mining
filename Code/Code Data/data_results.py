"""
BAQM Thesis 2021-2022 - Tip Mining
This file is used to do calculations with the result data of the BERT model.

@author: Erik van der Heide
"""

"""Plot distribution of predicted tips""" 
# Import packages
import pandas as pd  
import matplotlib.pyplot as plt

results_baby = pd.read_csv("results_baby.csv", sep="\t", header=0)

preds = results_baby.prediction

# Plot histogram of words tips
plt.xlabel('Predicted probability')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.hist(results_baby['prediction'], bins=40, color='#0504aa', alpha=0.8, 
         rwidth=0.84)
plt.locator_params(axis="x", nbins=20)
plt.show()

"""Calculate the Fleiss Kappa (as check of Excel calculation)"""
# fleiss kappa calculation
z = pd.read_csv("fleiss_kappa_table.csv", sep=";", header=None)
from statsmodels.stats.inter_rater import fleiss_kappa
fleiss_kappa(z)

"""Calculate most frequent unigrams of results"""
# Packages
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load results
results_baby = pd.read_csv("results_baby.csv", sep="\t", header=0)
results_cloth = pd.read_csv("results_cloth.csv", sep="\t", header=0)
results_food = pd.read_csv("results_food.csv", sep="\t", header=0)
results_health = pd.read_csv("results_health.csv", sep="\t", header=0)
results_music = pd.read_csv("results_music.csv", sep="\t", header=0)
results_phone = pd.read_csv("results_phone.csv", sep="\t", header=0)
results_sports = pd.read_csv("results_sports.csv", sep="\t", header=0)
results_tools = pd.read_csv("results_tools.csv", sep="\t", header=0)
results_toys = pd.read_csv("results_toys.csv", sep="\t", header=0)
results_video = pd.read_csv("results_video.csv", sep="\t", header=0)

results_baby_tips = results_baby[results_baby['prediction']>=0.5].sentence
results_cloth_tips = results_cloth[results_cloth['prediction']>=0.5].sentence
results_food_tips = results_food[results_food['prediction']>=0.5].sentence
results_health_tips = results_health[results_health['prediction']>=0.5].sentence
results_music_tips = results_music[results_music['prediction']>=0.5].sentence
results_phone_tips = results_phone[results_phone['prediction']>=0.5].sentence
results_sports_tips = results_sports[results_sports['prediction']>=0.5].sentence
results_tools_tips = results_tools[results_tools['prediction']>=0.5].sentence
results_toys_tips = results_toys[results_toys['prediction']>=0.5].sentence
results_video_tips = results_video[results_video['prediction']>=0.5].sentence

data_tips = ' '.join(pd.concat([results_baby_tips,
                                results_cloth_tips,
                                results_food_tips,
                                results_health_tips,
                                results_music_tips,
                                results_phone_tips,
                                results_sports_tips,
                                results_tools_tips,
                                results_toys_tips,
                                results_video_tips
                                ]))
text_tokens = word_tokenize(data_tips)

# Delete stopwords
big_text = [word for word in text_tokens if not word in stopwords.words()]

Counter = Counter(big_text)
  
# most_common() produces k frequently encountered
# input values and their respective counts.
most_occur = Counter.most_common(100)
  
print(most_occur)


#
text = "hello hi hey hai hoi hu ha"
import nltk
from nltk import word_tokenize 
from nltk.util import ngrams


