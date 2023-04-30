#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import pandas Library for Load the Dataset.
import pandas as pd


# In[2]:


# Load the dataset.
data = pd.read_csv("test.csv").head(200)
data


# In[3]:


# neattext.functions provides a useful set of tools for cleaning, normalizing, and standardizing text data
# which can be valuable for text analysis and natural language processing tasks.
import neattext.functions as nfx


# It is used to remove stopwords from a piece of text
# Stopwords are common words in a language that do not carry much meaning on their own.
# such as "the", "a", "an", "and", "in", "of", etc
data['tweet'] = data['tweet'].apply(nfx.remove_stopwords) 


# It is used to remove user handles or usernames from a piece of text.
# User handles or usernames are typically used in social media platforms such as Twitter and Instagram to mention or tag other users
data['tweet'] = data['tweet'].apply(nfx.remove_userhandles)


# It is used to remove punctuation marks from a piece of text.
# Punctuation marks include symbols such as periods(.) , commas(,) , question marks(?) , exclamation marks(!) , etc.
data['tweet'] = data['tweet'].apply(nfx.remove_punctuations)


# It is used to remove emojis from a piece of text.
data['tweet'] = data['tweet'].apply(nfx.remove_emojis)

# It is used to remove multiple spaces from a piece of text and replaces them with a single space.
data['tweet'] = data['tweet'].apply(nfx.remove_multiple_spaces)

# It is used to remove non-ASCII characters from a piece of text
data['tweet'] = data['tweet'].apply(nfx.remove_non_ascii)

# It method is used to remove punctuations from a piece of text.
data['tweet'] = data['tweet'].apply(nfx.remove_puncts)
data


# In[4]:


# The Natural Language Toolkit (nltk) is a popular Python library for working with human language data
# It provides a wide range of tools and resources for tasks such as tokenization, stemming, tagging.
# tokenization  :- It means convert text into list of words word_tokenize(),convert text into list of sentence sent_tokenize()
# stemming      :- Stemming is the process of reducing words to their base or root form.
# tagging       :- Tagging is the process of adding labels or tags to pieces of information to categorize them and make 
#                  them easier to find and organize
import nltk

# The re module in Python provides support for regular expressions (regex)
# The module is used to perform various text operations, such as pattern matching, search and replace, and text parsing.
import re

# It devide the text/sentence into list of words.
from nltk.tokenize import word_tokenize

# Store tweet column value into one string.
text_data = ' '.join(data['tweet'].astype(str))

# tokenize the text string into words
words = word_tokenize(text_data)

# Word for Hashtag
wd = text_data.split(" ")

# It is used to assigning a part of speech tag (noun, verb, adjective, adverb, etc.) to each word in a text.
pos_tags = nltk.pos_tag(words)

# initialize counts for each part of speech
verb_count = 0
adverb_count = 0
adjective_count = 0
noun_count = 0
pronoun_count = 0
hashtag_count = 0
number_count = 0
special_count = 0

# List of all Functionality.
verbs = []
adverbs = []
adjectives = []
nouns = []
pronouns = []
hashtags = []
special_simbols = []

# loop through the tagged words and count the instances of each part of speech and hashtags
for word,pos in pos_tags:
    if pos.startswith('V'):  # verbs (VBP)
        verb_count += 1
        verbs.append(word)
        
    elif pos.startswith('RB'):  # adverbs(RB)
        adverb_count += 1
        adverbs.append(word)
        
    elif pos.startswith('JJ'):  # adjectives (JJ)
        adjective_count += 1
        adjectives.append(word)
        
    elif pos.startswith('N'):  # nouns (NN)
        noun_count += 1
        nouns.append(word)
        
    elif pos.startswith('PR'):  # pronouns (PRP)
        pronoun_count += 1
        pronouns.append(word)
        
    # ^ circumflex means not,\s means matches any whitespace character (i.e., space, tab, newline, etc
    elif re.match('[^a-zA-Z0-9\s]+', word):  # special symbols
        special_count += 1
        special_simbols.append(word)


# hashtags    
for word in wd:
    if word.startswith('#'):  # hashtags
        hashtag_count += 1
        hashtags.append(word)
        
# Numbers
import re
numbers = re.findall(r'\d+',text_data)
number_count = len(numbers)

        
# # print the counts for each part of speech and hashtags
print('Verb count           :- ', verb_count)
print('Adverb count         :- ', adverb_count)
print('Adjective count      :- ', adjective_count)
print('Noun count           :- ', noun_count)
print('Pronoun count        :- ', pronoun_count)
print('Hashtag count        :- ', hashtag_count)
print('Number count         :- ', number_count)
print('Special symbol count :- ', special_count)



# In[5]:


print(verbs)


# In[6]:


print(adverbs)


# In[7]:


print(nouns)


# In[8]:


print(pronouns)


# In[9]:


print(hashtags)


# In[10]:


import numpy as np
print(np.unique(numbers))


# In[11]:


print(np.unique(special_simbols))


# In[ ]:




