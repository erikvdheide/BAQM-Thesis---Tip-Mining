"""
BAQM Thesis 2021 - Tip Mining
Plot the coverage of the data for different maximum sequence lengths.

@author: Erik van der Heide
"""

# Packages
import pandas as pd
import transformers as ppb
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

""" 
Pre-process data
"""

# Load data
df = pd.read_csv("td_clean.csv", sep="\t", header=0)
df = df[['sentence', 'tip']]
df.reset_index(drop=True, inplace=True)

# Set tokenezier
tokenizer_class = ppb.BertTokenizer
pretrained_weights = 'bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

# Tokenize: break up in words/tokens (101 = [CLS] token, 102 = [SEP] token)
tokenized = df["sentence"].apply((lambda x: tokenizer.encode(x)))
print("Example tokenized sentence: ", tokenized[0], "\n")

"""
Plots of sequence length
"""
tokenized_length = []
cov_16 = 0
cov_32 = 0
cov_64 = 0
cov_128 = 0
for i in tokenized.values:
    tokenized_length.append(len(i))
    if len(i)<=16:
        cov_16 = cov_16 + 1
    if len(i)<=32:
        cov_32 = cov_32 + 1
    if len(i)<=64:
        cov_64 = cov_64 + 1
    if len(i)<=128:
        cov_128 = cov_128 + 1
print("\n<= 16:", cov_16 / len(tokenized_length))
print("<= 32:", cov_32 / len(tokenized_length))
print("<= 64:", cov_64 / len(tokenized_length))
print("<= 128:", cov_128 / len(tokenized_length))
del cov_16, cov_32, cov_64, cov_128

# Descriptive stats
print("\nMin: ", min(tokenized_length))
print("Mean: ", sum(tokenized_length) / len(tokenized_length))
print("Max: ", max(tokenized_length))

# Histogram 
plt.xlabel('Num. of tokens in training dataset')
plt.ylabel('Frequency')
plt.xlim(right=100)
plt.grid(axis='y', alpha=0.75)
plt.hist(tokenized_length, bins=75, color='#0504aa', alpha=0.8, rwidth=0.85)

# Empirical CDF
ecdf = ECDF(tokenized_length)
print('\nP(x<16): %.3f' % ecdf(16))
print('P(x<32): %.3f' % ecdf(32))
print('P(x<64): %.3f' % ecdf(64))
print('P(x<128): %.3f' % ecdf(128))
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Num. of tokens in training dataset')
plt.ylabel('Coverage')
plt.plot(ecdf.x, ecdf.y, color='#0504aa')
plt.axvline(x=16, linestyle='--', color='black', alpha=0.3)
plt.axvline(x=32, linestyle='--', color='black', alpha=0.3)
plt.axvline(x=64, linestyle='--', color='black', alpha=0.3)
plt.axvline(x=128, linestyle='--', color='black', alpha=0.3)
plt.show()