"""
BAQM Thesis 2021-2022 - Tip Mining
This file contains data preprocessing of full data and constructs evaluation data.

@author: Erik van der Heide
"""
# Import packages
import pandas as pd
import gzip
import nltk
import time
import random

# Silence SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)

"""
Define functions to read the data
"""
# Reading in the data
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

# Read data into pandas data frame        
def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')    
    
"""
Reading data of 5 main categories
"""
# Read datasets (df = DataFrame)
df_baby = getDF("INPUT DATA/reviews_Baby_5.json.gz")
df_music = getDF("INPUT DATA/reviews_Musical_Instruments_5.json.gz")
df_sports = getDF("INPUT DATA/reviews_Sports_and_Outdoors_5.json.gz")
df_tools = getDF("INPUT DATA/reviews_Tools_and_Home_Improvement_5.json.gz")
df_toys = getDF("INPUT DATA/reviews_Toys_and_Games_5.json.gz")

df_baby['category'] = "Baby"
df_music['category'] = "Music & Instruments"
df_sports['category'] = "Sports & Outdoors"
df_tools['category'] = "Tools & Home Improvement"
df_toys['category'] = "Toys & Games"

#for dfm in [df_baby, df_music, df_sports, df_tools, df_toys]:
#    print("\nCategory         :", dfm.name)
#    print("Num observations :", dfm.shape[0])
#    print("Num products     :", dfm['asin'].nunique())

# Sentence tokenizer (sentence separator)
nltk.sent_tokenize("Hello! World! Have a nice day.")

"""
Create data to be evaluated 
"""
# SUBSET PARAMETERS
num_prod = 10   # num of products for each category to include in evaluation data 
num_reviews = 5 # num of reviews per product in the evaluation data

def dataPreprocessing(df, tdf):
    """Function that does data preprocessing:
        - split reviews into sentences 
        - delete sentences of less than two words
    Args:
        df = DataFrame from one of the cateogires
        tdf = corresponding TrainingDataFrame (0 if other category)
    """
    
    # Alter columns to get them in the same style
    df['reviewer_id'] = df['reviewerID']
    df['review_text'] = df['reviewText']
    df.drop(['reviewerName','helpful','overall','summary','unixReviewTime', 
              'reviewerID','reviewTime','reviewText'], inplace=True, axis=1)
    df = df.reindex(columns=['asin','reviewer_id','category', 'review_text'])
    ev = df.copy() # evaluation data      
    
    # Only for main categories, delete products from training data 
    if len(tdf)>1: 
        prods_in_tf = tdf['asin'].unique()
        ev = ev[~ev['asin'].isin(prods_in_tf)]
        ev.reset_index(drop=True, inplace=True)
        del prods_in_tf
    
    # TODO zoek uit hoe aantal prod niet synchroom loopt
    print("\nCategory           :", df['category'][0])
    print("Num products in df :", df['asin'].nunique())
    if len(tdf)>1: print("Num products in tf :", tdf['asin'].nunique())
    print("Num products in ev :", ev['asin'].nunique())
    print("Num reviews in df  :", df.shape[0])   
    if len(tdf)>1: print("Num reviews in tf  :", len(tdf['review_id'].unique()))
    print("Num reviews in ev  :", ev.shape[0])
    
    # TODO from here figure out how code works
    # Subset num_prod random products
    if num_prod > 0:
        prods = ev['asin'].unique()
        random.seed(2)
        ran_num = random.sample(range(len(prods)), num_prod) # sample num_prod product indices
        prods_sub = [prods[i] for i in ran_num] # choose num_prod random products
        ev = ev[ev['asin'].isin(prods_sub)]        
        
    # Sentence-tokenize to be evaluated reviews
    ev_tok = ev
    s = ev_tok["review_text"].apply(lambda x : nltk.sent_tokenize(x)).apply(pd.Series,1).stack()
    s.index = s.index.droplevel(-1)
    s.name = 'sentence'
    del ev_tok['review_text']
    ev_tok = ev_tok.join(s)
    del s
    ev_tok.reset_index(drop=True, inplace=True)

    # Calculate sentence number
    ev_tok['num_review'] = 0
    ev_tok['num_review'][0] = 1
    ev_tok['num_sentence'] = 0
    ev_tok['num_sentence'][0] = 1
    counter = 1
    counter_rev = 1
    for index, row in ev_tok.iterrows():
        if (index > 0):
            if (ev_tok['reviewer_id'][index] == 
                ev_tok['reviewer_id'][index-1]):
                counter = counter + 1
                ev_tok['num_sentence'][index] = counter
                ev_tok['num_review'][index] = counter_rev
            else:
                counter = 1
                counter_rev = counter_rev + 1
                ev_tok['num_sentence'][index] = counter
                ev_tok['num_review'][index] = counter_rev
    del counter, index
        
    # Delete sentences with less than 2 words 
    ev_tok['sen_length'] = 0
    for index, row in ev_tok.iterrows():
        tokens = nltk.word_tokenize(str(ev_tok['sentence'][index]))
        words = [word for word in tokens if word.isalpha()]
        ev_tok['sen_length'][index] = len(words)
    del index, row, tokens, words
    ev_tok_clean = ev_tok[ev_tok["sen_length"] >= 2]
    ev_tok_clean.reset_index(drop=True, inplace=True)
    print("Reduced sentences from too small sentences: ", 
          ev_tok.shape[0]-ev_tok_clean.shape[0], " (", round(100*(
          ev_tok_clean.shape[0]-ev_tok.shape[0])/ev_tok.shape[0], 2), "%)")
    
    # TODO replace this by something easier (below code which is above?)
    # Keep 5 reviews per product
    #if num_reviews > 0:
    #    reviewNumbers = []
    #    prods = ev['asin'].unique()
    #    random.seed(2)
    #    for prod in prods:
    #        ev_temp = ev_tok[ev_tok['asin']==prod]
    #        review_numbers = ev_temp['num_review'].unique()
    #        ran_num2 = random.sample(range(len(review_numbers)), num_reviews) # sample 5 reviews numbers
    #        reviewers_sub = [review_numbers[i] for i in ran_num2] # choose 10 random products
    #        reviewNumbers.append(reviewers_sub)
    #    ev_tok = ev_tok[ev_tok['num_review'].isin(reviewNumbers)]   

    return ev_tok_clean

def keep5reviews(df):
    keep_reviews = [1,2,3,4,5]
    for index, row in df.iterrows():
        if index > 0:
            if df['asin'][index] != df['asin'][index-1]:
                strt = df['num_review'][index]
                keep_reviews.append(strt)
                keep_reviews.append(strt+1)
                keep_reviews.append(strt+2)
                keep_reviews.append(strt+3)
                keep_reviews.append(strt+4)
    df = df[df['num_review'].isin(keep_reviews)]
    return df    

"""
Run data preprocessing for 5 main categories
"""

# BABY (Running time: 0.92 min)
start = time.time()
ev_baby = dataPreprocessing(df_baby, td_baby)
end = time.time()
print("Elapsed time Baby: ", (end-start)/60, "min\n")
del start, end

# MUSIC (0.004 min for 90th percentile)
start = time.time()
ev_music = dataPreprocessing(df_music, td_music)
end = time.time()
print("Elapsed time Music: ", (end-start)/60, "min\n")
del start, end

# SPORTS (2.78 min for 90th percentile)
start = time.time()
ev_sports = dataPreprocessing(df_sports, td_sports)
end = time.time()
print("Elapsed time Sports: ", (end-start)/60, "min\n")
del start, end

# TOOLS (1.11 min for 90th percentile)
start = time.time()
ev_tools = dataPreprocessing(df_tools, td_tools)
end = time.time()
print("Elapsed time Tools: ", (end-start)/60, "min\n")
del start, end

# TOYS (2.02 min for 90th percentile)
start = time.time()
ev_toys = dataPreprocessing(df_toys, td_toys)
end = time.time()
print("Elapsed time Toys: ", (end-start)/60, "min\n")
del start, end           

# Keep 5 reviews per category
ev_baby_5 = keep5reviews(ev_baby)
ev_music_5 = keep5reviews(ev_music)
ev_sports_5 = keep5reviews(ev_sports)
ev_tools_5 = keep5reviews(ev_tools)
ev_toys_5 = keep5reviews(ev_toys)

# Write evaluation data to csv
ev_baby.to_csv("ev10_baby.csv", sep = "\t")
ev_music.to_csv("ev10_music.csv", sep = "\t")
ev_sports.to_csv("ev10_sports.csv", sep = "\t")
ev_tools.to_csv("ev10_tools.csv", sep = "\t")
ev_toys.to_csv("ev10_toys.csv", sep = "\t")

ev_baby_5.to_csv("ev10_baby_5.csv", sep = "\t")
ev_music_5.to_csv("ev10_music_5.csv", sep = "\t")
ev_sports_5.to_csv("ev10_sports_5.csv", sep = "\t")
ev_tools_5.to_csv("ev10_tools_5.csv", sep = "\t")
ev_toys_5.to_csv("ev10_toys_5.csv", sep = "\t")


##############################################################################
#                 5 NEW CATEGORIES WITHOUT TD                                #
##############################################################################

"""
Reading data of 5 extra categories
"""
# Read datasets (df = DataFrame)
df_cloth = getDF("reviews_Clothing_Shoes_and_Jewelry_5.json.gz")
df_phone = getDF("reviews_Cell_Phones_and_Accessories_5.json.gz")
df_health = getDF("reviews_Health_and_Personal_Care_5.json.gz")
df_video = getDF("reviews_Video_Games_5.json.gz")
df_food = getDF("reviews_Grocery_and_Gourmet_Food_5.json.gz")

print("\nCloth: Num. observations:", df_cloth.shape[0], 
          "- Num products:", df_cloth['asin'].nunique(), "\n")
print("Phone: Num. observations: ", df_phone.shape[0],
           "- Num products:", df_phone['asin'].nunique(), "\n")
print("Health: Num. observations: ", df_health.shape[0],
            "- Num products:", df_health['asin'].nunique(), "\n")
print("Video: Num. observations: ", df_video.shape[0],
           "- Num products:", df_video['asin'].nunique(), "\n")
print("Food: Num. observations: ", df_food.shape[0],
          "- Num products:", df_food['asin'].nunique(), "\n")

# Add category name to datasets
df_cloth['category'] = "Clothing, Shoes & Jewelry"
df_phone['category'] = "Cell Phones & Accessories"
df_health['category'] = "Health & Personal Care"
df_video['category'] = "Video Games"
df_food['category'] = "Grocery & Gourmet Food"
    
# Evaluate products from other categories (tdf = 0)
ev_cloth = dataPreprocessing(df_cloth, [0])
ev_phone = dataPreprocessing(df_phone, [0])
ev_health = dataPreprocessing(df_health, [0])
ev_video = dataPreprocessing(df_video, [0])
ev_food = dataPreprocessing(df_food, [0])

# Keep 5 reviews per category
ev_cloth_5 = keep5reviews(ev_cloth)
ev_phone_5 = keep5reviews(ev_phone)
ev_health_5 = keep5reviews(ev_health)
ev_video_5 = keep5reviews(ev_video)
ev_food_5 = keep5reviews(ev_food)

# Write evaluation data to csv
ev_cloth.to_csv("ev10_cloth.csv", sep = "\t")
ev_phone.to_csv("ev10_phone.csv", sep = "\t")
ev_health.to_csv("ev10_health.csv", sep = "\t")
ev_video.to_csv("ev10_video.csv", sep = "\t")
ev_food.to_csv("ev10_food.csv", sep = "\t")

ev_cloth_5.to_csv("ev10_cloth_5.csv", sep = "\t")
ev_phone_5.to_csv("ev10_phone_5.csv", sep = "\t")
ev_health_5.to_csv("ev10_health_5.csv", sep = "\t")
ev_video_5.to_csv("ev10_video_5.csv", sep = "\t")
ev_food_5.to_csv("ev10_food_5.csv", sep = "\t")
