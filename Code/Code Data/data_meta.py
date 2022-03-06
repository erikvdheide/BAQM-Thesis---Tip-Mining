"""
BAQM Thesis 2021-2022 - Tip Mining
This file is used to get the name of the products in the evaluation data.

@author: Erik van der Heide
"""

import numpy as np
import pandas as pd
import gzip
np.set_printoptions(threshold=np.inf)

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
meta_baby = getDF("meta_Baby.json.gz")
meta_cloth = getDF("meta_Clothing_Shoes_and_Jewelry.json.gz")
meta_food = getDF("meta_Grocery_and_Gourmet_Food.json.gz")
meta_health = getDF("meta_Health_and_Personal_Care.json.gz")
meta_music = getDF("meta_Musical_Instruments.json.gz")
meta_phone = getDF("meta_Cell_Phones_and_Accessories.json.gz")
meta_sports = getDF("meta_Sports_and_Outdoors.json.gz")
meta_tools = getDF("meta_Tools_and_Home_Improvement.json.gz")
meta_toys = getDF("meta_Toys_and_Games.json.gz")
meta_video = getDF("meta_Video_Games.json.gz")

# Baby
meta_baby[meta_baby['asin']=='B000DZTS2I'].title
meta_baby[meta_baby['asin']=='B000JNP0F8'].title
meta_baby[meta_baby['asin']=='B000VKCGTM'].title
meta_baby[meta_baby['asin']=='B000YDH4HA'].title
meta_baby[meta_baby['asin']=='B001UFBVA2'].title
meta_baby[meta_baby['asin']=='B002A9JCY4'].title
meta_baby[meta_baby['asin']=='B002WZ2RE8'].title
meta_baby[meta_baby['asin']=='B003KGANTO'].title
meta_baby[meta_baby['asin']=='B004H69GSU'].title
meta_baby[meta_baby['asin']=='B005CX0208'].title

# Clothes
meta_cloth[meta_cloth['asin']=='B000J3H31W'].title
meta_cloth[meta_cloth['asin']=='B000UPVCPW'].title
meta_cloth[meta_cloth['asin']=='B000W8YCHW'].title
meta_cloth[meta_cloth['asin']=='B002BA5NS6'].title
meta_cloth[meta_cloth['asin']=='B00366DZZM'].title
meta_cloth[meta_cloth['asin']=='B003VPOBCO'].title
meta_cloth[meta_cloth['asin']=='B004NNVGNA'].title
meta_cloth[meta_cloth['asin']=='B005CPQRD2'].title
meta_cloth[meta_cloth['asin']=='B00B4KHI5A'].title
meta_cloth[meta_cloth['asin']=='B00E9IULVC'].title

# Food
meta_food[meta_food['asin']=='B000E1BL5S'].title
meta_food[meta_food['asin']=='B000ER1DDW'].title
meta_food[meta_food['asin']=='B000FYYOXU'].title
meta_food[meta_food['asin']=='B000GZU7QQ'].title
meta_food[meta_food['asin']=='B00121BQJU'].title
meta_food[meta_food['asin']=='B001652KD8'].title
meta_food[meta_food['asin']=='B001EO76U8'].title
meta_food[meta_food['asin']=='B001PB2ZP6'].title
meta_food[meta_food['asin']=='B002ZYYUGE'].title
meta_food[meta_food['asin']=='B0044R3DNG'].title

# Health
meta_health[meta_health['asin']=='B0002DSVTC'].title 
meta_health[meta_health['asin']=='B0009R5AA4'].title
meta_health[meta_health['asin']=='B000FSZNZY'].title
meta_health[meta_health['asin']=='B000GFPBIK'].title
meta_health[meta_health['asin']=='B000YDJIYW'].title
meta_health[meta_health['asin']=='B0012BSOB8'].title
meta_health[meta_health['asin']=='B001DB6XWE'].title
meta_health[meta_health['asin']=='B001MIZMIE'].title
meta_health[meta_health['asin']=='B002UZQT58'].title
meta_health[meta_health['asin']=='B003Z0IPXG'].title

# Music
meta_music[meta_music['asin']=='B00074B67A'].title
meta_music[meta_music['asin']=='B000A1HU1G'].title
meta_music[meta_music['asin']=='B000AAGM0M'].title
meta_music[meta_music['asin']=='B000LAT0AK'].title
meta_music[meta_music['asin']=='B000RY68PA'].title
meta_music[meta_music['asin']=='B000WN4J9S'].title
meta_music[meta_music['asin']=='B001E3SFKO'].title
meta_music[meta_music['asin']=='B001Q8DJO4'].title
meta_music[meta_music['asin']=='B007T8CUNG'].title
meta_music[meta_music['asin']=='B00BTGMI5O'].title

# Phone
meta_phone[meta_phone['asin']=='B0036255ZE'].title
meta_phone[meta_phone['asin']=='B003Y5A9HM'].title
meta_phone[meta_phone['asin']=='B0040PDK0I'].title
meta_phone[meta_phone['asin']=='B00574S96G'].title
meta_phone[meta_phone['asin']=='B005XHVWGQ'].title
meta_phone[meta_phone['asin']=='B006WIED5M'].title
meta_phone[meta_phone['asin']=='B0087Y3BLG'].title
meta_phone[meta_phone['asin']=='B0092QSQ3Q'].title
meta_phone[meta_phone['asin']=='B00HJKRQAQ'].title
meta_phone[meta_phone['asin']=='B00HPMB38Y'].title

# Sports
meta_sports[meta_sports['asin']=='B000AR2N76'].title
meta_sports[meta_sports['asin']=='B000F38SZ6'].title
meta_sports[meta_sports['asin']=='B000FH1E0I'].title
meta_sports[meta_sports['asin']=='B000QJAFIM'].title
meta_sports[meta_sports['asin']=='B001UXQXSE'].title
meta_sports[meta_sports['asin']=='B0061PWBLY'].title
meta_sports[meta_sports['asin']=='B008NEFDK2'].title
meta_sports[meta_sports['asin']=='B0093IKQT0'].title
meta_sports[meta_sports['asin']=='B009SM5IR6'].title
meta_sports[meta_sports['asin']=='B00I0HQ4MS'].title

# Tools
meta_tools[meta_tools['asin']=='B00004Z0YC'].title
meta_tools[meta_tools['asin']=='B00008BFS5'].title
meta_tools[meta_tools['asin']=='B00009K77A'].title
meta_tools[meta_tools['asin']=='B0006SU3QW'].description
meta_tools[meta_tools['asin']=='B000PFNCHI'].title
meta_tools[meta_tools['asin']=='B0052MG5K0'].title
meta_tools[meta_tools['asin']=='B007QV4PZM'].title
meta_tools[meta_tools['asin']=='B008UPDPIG'].title
meta_tools[meta_tools['asin']=='B009KZ20D6'].title
meta_tools[meta_tools['asin']=='B00ANI1QXY'].title

# Toys
meta_toys[meta_toys['asin']=='B0000C9WI2'].title
meta_toys[meta_toys['asin']=='B0007CSEYA'].title
meta_toys[meta_toys['asin']=='B000F6RWW8'].title
meta_toys[meta_toys['asin']=='B000GKAU0O'].title
meta_toys[meta_toys['asin']=='B001A5SNXA'].description
meta_toys[meta_toys['asin']=='B001GZXYFG'].title
meta_toys[meta_toys['asin']=='B002P584A6'].title
meta_toys[meta_toys['asin']=='B003BNZNNW'].title
meta_toys[meta_toys['asin']=='B004JA7M4E'].title
meta_toys[meta_toys['asin']=='B005KSZO0I'].title

# Video
meta_video[meta_video['asin']=='B00004WHW7'].description
meta_video[meta_video['asin']=='B00005Q8KY'].description
meta_video[meta_video['asin']=='B00005V3FA'].description
meta_video[meta_video['asin']=='B0002FQVEW'].description
meta_video[meta_video['asin']=='B000ASDU40'].description
meta_video[meta_video['asin']=='B000HNJ5WE'].description
meta_video[meta_video['asin']=='B000WJOZAK'].description
meta_video[meta_video['asin']=='B001E8VB6O'].description
meta_video[meta_video['asin']=='B00AMQMMEO'].description
meta_video[meta_video['asin']=='B00AQ6FSLY'].description

