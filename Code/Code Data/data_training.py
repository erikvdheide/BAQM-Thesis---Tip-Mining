"""
BAQM Thesis 2021-2022 - Tip Mining
This file contains data preprocessing and exploration of the training data.

@author: Erik van der Heide
"""

# Import packages
import pandas as pd  
import nltk      
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS

# Silence SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)

# Download training data (td = TrainingData)
# path = 'C:\Users\erikv\BAQM Master Thesis Spyder\INPUT DATA\generate_tips_data.tsv'
td = pd.read_csv(r'C:\Users\erikv\BAQM Master Thesis Spyder\INPUT DATA\generate_tips_data_fixed.tsv', sep="\t", header=0)
# td = pd.read_csv("INPUT DATA/generate_tips_data_fixed.tsv", sep="\t", header=0)

##############################################################################
#                         Data Preprocessing                                 #
##############################################################################

# Correct few mistakes in the data
td.dtypes
td.isna().sum() # 2 tips are still NA
td['tip'] = td['tip'].fillna(0)
td['sentence'][2258] = td['sentence'][2258].rsplit(' ', 1)[0]
td['sentence'][69493] = td['sentence'][69493].rsplit(' ', 1)[0]

print("\nFirst data insights:")
print("Raw total num sentences :", td.shape[0])
print("Raw tips in data        :", td[td['tip']==1].shape[0])
print("Raw non-tips in data    :", td[td['tip']==0].shape[0])
print("Raw tip ratio in data   :", round(100*td[td['tip']==1].shape[0]/td.shape[0],2))

# Word tokenizer (word separator)
nltk.word_tokenize("This. is to TEST tokenization. :)")

# Find number of words per sentence
td['sen_length'] = 0
for index, row in td.iterrows():
    tokens = nltk.word_tokenize(str(td['sentence'][index]))
    words = [word for word in tokens if word.isalpha()]
    td['sen_length'][index] = len(words)
    ## OPTIONAL BELOW: delete punctuation
    #td['sentence'][index] = ' '.join(word for word in words) # Del punctuation
del index, row, tokens, words

##############################################################################

"""
td_clean: Delete sentences with one of zero words
"""
td_clean = td[td["sen_length"] >= 2]
td_clean.reset_index(drop=True, inplace=True) # 84071 sentences remaining
print("\nDeleting all sentences with less than 2 words:")
print("Reduced sentences     :", td.shape[0]-td_clean.shape[0])
print("Reduced sentences (%) :", round(100*(td_clean.shape[0]-td.shape[0])/td.shape[0], 2))
print("Num of tips lost      :", td[td["tip"]==1].shape[0]-td_clean[td_clean["tip"]==1].shape[0])
print("Num tips in total     :", td_clean[td_clean['tip']==1].shape[0])
print("Num non-tips in total :", td_clean[td_clean['tip']==0].shape[0])
print("Tip ratio             :", round(100*td_clean[td_clean['tip']==1].shape[0]/td_clean.shape[0],2))


##############################################################################

"""
Subsetting the data on the 5 main categories
"""
td_baby = td_clean[td_clean['category'] == "Baby"]
td_music = td_clean[td_clean['category'] == "Musical Instruments"]
td_sports = td_clean[td_clean['category'] == "Sports & Outdoors"]
td_tools = td_clean[td_clean['category'] == "Tools & Home Improvement"]
td_toys = td_clean[td_clean['category'] == "Toys & Games"]

for tdf in [td_baby, td_music, td_sports, td_tools, td_toys]:
    tdf.reset_index(drop=True, inplace=True)
    print("\nCategory      :", tdf['category'][0])
    print("Num tips      :", tdf[tdf['tip']==1].shape[0])
    print("Num non-tips  :", tdf[tdf['tip']==0].shape[0])
    print("Num sentences :", tdf.shape[0])
    print("Tip ratio     :", round(100*tdf[tdf['tip']==1].shape[0]/tdf.shape[0],2))
    print("Num products  :", len(set(tdf['asin'])), "\n")

##############################################################################
#                           Data exploration                                 #
##############################################################################

"""
Investigate tips
"""
# Subset on tips
td_tips = td_clean[td_clean['tip']==1]

# Wordcloud exluding stopwords
text = " ".join(td_tips['sentence'])
stop_words = set(STOPWORDS)
wordcloud_tips = WordCloud(width=2000, height=2000, background_color='white', 
                           stopwords=stop_words, min_font_size=20, 
                           max_font_size=400).generate(text)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud_tips)
plt.axis("off")
plt.show()
del stop_words, wordcloud_tips, text

# Statistics of word length of tips
td_tips['sen_length'].describe()
distribution_tips = {i:list(td_tips['sen_length']).count(i) 
                     for i in set(list(td_tips['sen_length']))}
print("\nDistribution tips: \n", distribution_tips)

# Plot histogram of words tips
plt.xlabel('Num. of words in tip')
plt.ylabel('Tip frequency')
plt.grid(axis='y', alpha=0.75)
_ = plt.hist(td_tips['sen_length'], bins=50, color='#0504aa', alpha=0.8, rwidth=0.84)

# Print short tips
print("\nShort tips: ")
for index, row in td_tips.iterrows():
    if td_tips['sen_length'][index] == 2:
        print(td_tips['sentence'][index])
del index, row
del distribution_tips

"""
Investigate non-tips
"""
# Subset on non-tips
td_nontips = td_clean[td_clean["tip"]==0]

# Statistics of word length of non-tips
td_nontips['sen_length'].describe()
distribution_nontips = {i:list(td_nontips["sen_length"]).count(i) 
                        for i in set(list(td_nontips["sen_length"]))}
print("\nDistribution non-tips: \n", distribution_nontips)
del distribution_nontips

# Plot histogram of words non-tips
plt.xlabel('Num. of words in non-tip')
plt.ylabel('Non-tip frequency')
plt.grid(axis='y', alpha=0.75)
_ = plt.hist(td_nontips["sen_length"], bins=50, color='#0504aa', alpha=0.8, rwidth=0.85)

##############################################################################
#  Optional addition: Rule based learning to reduce unbalanced dataset size  #
##############################################################################

def rule1(data):
    "Rule 1: delete senteneces of 5 words and fewer"
    data = data[data['sen_length']>=6]
    data.reset_index(drop=True, inplace=True)
    return data

def rule2(data):
    "Rule 2: delete sentences with enthusiastic words"
    ent_words = ["wonderful", "adorable", "amazing", "fantastic", "love", "like"]
    data['rule2'] = 0
    for index, row in data.iterrows():
        tokens = nltk.tokenize.word_tokenize(str(data["sentence"][index]))
        for token in tokens:
            if token in ent_words:
                data['rule2'][index] = 1
    data = data[data['rule2']==0]
    data.reset_index(drop=True, inplace=True)
    return data

def rule3(data):
    "Rule 3: delete sentences about price, shipping and return policy"
    list_words = ["price", "money", "cheap", "expensive", "shipping", "return", 
                  "warranty", "$", "€"]
    data['rule3'] = 0
    for index, row in data.iterrows():
        tokens = nltk.tokenize.word_tokenize(str(data["sentence"][index]))
        for token in tokens:
            if token in list_words:
                data['rule3'][index] = 1
    data = data[data['rule3']==0]
    data.reset_index(drop=True, inplace=True)
    return data
    
def rule4(data):
    "Rule 4: delete sentence with first-person pronoun"
    #pers_words = ["I'm", "I'll", "I've"]
    data['rule4'] = 0
    for index, row in data.iterrows():
        tokens = nltk.tokenize.word_tokenize(str(data["sentence"][index]))
        for i in range(0, len(tokens)-1):
            if (tokens[i]=="I" and tokens[i+1]=="'m") or (tokens[i]=="I" 
                and tokens[i+1]=="'ll") or (tokens[i]=="I" and tokens[i+1] 
                =="'ve") or tokens[i]=="Im" or tokens[i]=="Ill" or tokens[i]=="Ive":
                data['rule4'][index] = 1
    data = data[data['rule4']==0]
    data.reset_index(drop=True, inplace=True)
    return data
    
# Perform rules individually and together
td_1 = rule1(td)
td_2 = rule2(td)
td_3 = rule3(td)
td_4 = rule4(td)
td_clean_filtered = td.query('sen_length>=6 and rule2==0 and rule3==0 and rule4==0')
td_clean_filtered.drop(['rule2', 'rule3', 'rule4'], inplace=True, axis=1)

ind = 1
for td_rule in [td_1, td_2, td_3, td_4, td_clean_filtered]:
    if ind == 5:
       print("\nALL 4 RULES: ")
    print("\nReduced sentences in rule", ind, ":", td.shape[0]-td_rule.shape[0], 
          " (",round(100*(td_rule.shape[0]-td.shape[0])/td.shape[0], 1), "%)")
    print("Number of tips lost in rule", ind, ":", 
          td[td["tip"]==1].shape[0] - td_rule[td_rule["tip"]==1].shape[0],
          "(",round(100*(td_rule[td_rule["tip"]==1].shape[0]-td[td["tip"]==1].shape[0])/td[td["tip"]==1].shape[0],1),"%)")
    ind = ind+1
del ind

##############################################################################

"""
Use this code IF you want fixed train and test data!

# td_clean2: Subset data on products with at least 1 tip
prod_all = td_clean.asin.unique().shape[0]
tip_prods = td_clean[td_clean['tip']==1].asin.unique()
prod_tips = tip_prods.shape[0]
td_clean2 = td_clean[td_clean['asin'].isin(tip_prods)] # prods with >= 1 tip
clean2_tips = td_clean2[td_clean2['tip']==1].shape[0]
clean2_nontips = td_clean2[td_clean2['tip']==0].shape[0]
print("\nDeleting all sentences for products without tips in the data:")
print("Total num of products         :", prod_all)
print("Total num of products >=1 tip :", prod_tips)
print("%age products >=1 tip         :", round(100*prod_tips/prod_all,2))
print("Num tips cleaned data         :", clean2_tips)
print("Num non-tips cleaned data     :", clean2_nontips)
print("Tip ratio cleaned data        :", round(100*clean2_tips/td_clean2.shape[0],2))
del prod_all, prod_tips, clean2_tips, clean2_nontips, tip_prods

# Split data into 80% training and 20% testing for each category
td_baby_train, td_baby_test = train_test_split(td_baby, test_size=0.2, random_state=2021, stratify=td_baby['tip'])
td_music_train, td_music_test = train_test_split(td_music, test_size=0.2, random_state=2021, stratify=td_music['tip'])
td_sports_train, td_sports_test = train_test_split(td_sports, test_size=0.2, random_state=2021, stratify=td_sports['tip'])
td_tools_train, td_tools_test = train_test_split(td_tools, test_size=0.2, random_state=2021, stratify=td_tools['tip'])
td_toys_train, td_toys_test = train_test_split(td_toys, test_size=0.2, random_state=2021, stratify=td_toys['tip'])

# Concatenate train and test sets from categories into pooled sets
td_train = pd.concat([td_baby_train, td_music_train, td_sports_train, td_tools_train, td_toys_train])
td_test = pd.concat([td_baby_test, td_music_test, td_sports_test, td_tools_test, td_toys_test])
td_train.reset_index(drop=True, inplace=True)
td_test.reset_index(drop=True, inplace=True)

print("\nTrain and test splits:")
print("Size of train set (%) :", round(100*td_train.shape[0]/td_clean2.shape[0],2))
print("Num tips train set    :", td_train[td_train['tip']==1].shape[0])
print("Size of test set (%)  :", round(100*td_test.shape[0]/td_clean2.shape[0],2))
print("Num tips test set     :", td_test[td_test['tip']==1].shape[0])

# Write data to CSV
td_baby_train.to_csv("OUTPUT DATA/td_baby_train.csv", sep = "\t")
td_baby_test.to_csv("OUTPUT DATA/td_baby_test.csv", sep = "\t")
td_music_train.to_csv("OUTPUT DATA/td_music_train.csv", sep = "\t")
td_music_test.to_csv("OUTPUT DATA/td_music_test.csv", sep = "\t")
td_sports_train.to_csv("OUTPUT DATA/td_sports_train.csv", sep = "\t")
td_sports_test.to_csv("OUTPUT DATA/td_sports_test.csv", sep = "\t")
td_tools_train.to_csv("OUTPUT DATA/td_tools_train.csv", sep = "\t")
td_tools_test.to_csv("OUTPUT DATA/td_tools_test.csv", sep = "\t")
td_toys_train.to_csv("OUTPUT DATA/td_toys_train.csv", sep = "\t")
td_toys_test.to_csv("OUTPUT DATA/td_toys_test.csv", sep = "\t")
td_train.to_csv("OUTPUT DATA/td_train.csv", sep = "\t")
td_test.to_csv("OUTPUT DATA/td_test.csv", sep = "\t")
"""
