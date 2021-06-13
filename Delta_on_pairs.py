#!/usr/bin/env python
# coding: utf-8

# In[1]:


#packages
import csv
import numpy as np
import pandas as pd
from conllu import parse
from conllu import parse_incr
import os
import re
import nltk
nltk.download('punkt')


# In[3]:


#LOOP IT
set_nr = 0
nr_words = 300
foldernames = np.array(['before0','before1','after0','after1'])
mfwnames = np.array(['before0','after0'])
folder_with_pairs = ''
functionTags = set(["DT", "HA", "HD", "HP", "HS", "IE", "KN", "PL", "PN", "PP", "PS", "SN", "AB" ,"MAD","MID","PAD"])
        
burrow_file_name = folder_with_pairs + '_Burrows' + '.tsv'
with open(burrow_file_name, 'a', newline='') as out_file: #You can also include things like dates here
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['set','before','after','int_times','0_sen_count_before','1_sen_count_before','0_sen_count_after',
                             '1_sen_count_after']) 
cosine_file_name = folder_with_pairs  + '_Cosine' + '.tsv'
with open(cosine_file_name, 'a', newline='') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['set','before','after','int_times','0_sen_count_before','1_sen_count_before','0_sen_count_after',
                             '1_sen_count_after'])

def delta_for_multiple(dir_nr,set_nr):
    dict_with_text = {}
    #dict_with_pos = {}
    for filename in foldernames:
        textlist = []
        poslist = []
        data_file = open(folder_with_pairs + '/' + dir_nr + '/' + filename + '.conllu', "r", encoding="utf-8")
        for tokenlist in parse_incr(data_file):
            for i in range((len(tokenlist))):
                if tokenlist[i]['upos'] in functionTags:
                    if tokenlist[i]['upos'] == "AB" and tokenlist[i]['xpos'] != "AB":
                        continue
                    textlist.append(tokenlist[i]['form'])
                #poslist.append(tokenlist[i]['upos'])
        dict_with_text[filename] = textlist
        #dict_with_pos[filename] = poslist
        data_file.close()

    #--

    dict_with_text_tokens = {}
    #for text in foldernames:
        #tokens = nltk.word_tokenize(dict_with_text[text][0])
        #dict_with_text_tokens[text] = ([token for token in tokens
                                                #if any(c.isalpha() for c in token)])
    #dict_with_text_tokens = dict_with_pos  

    for text in foldernames:
        dict_with_text_tokens[text] = (
            [tok.lower() for tok in dict_with_text[text]])

    all_text = []
    for text in mfwnames:
        all_text += dict_with_text_tokens[text]

    all_text_freq_dist = list(nltk.FreqDist(all_text).most_common(nr_words))

    features = [word for word,freq in all_text_freq_dist]
    feature_freqs = {}

    for text in foldernames:
        feature_freqs[text] = {}

        overall = len(dict_with_text_tokens[text])

        for feature in features:
            presence = dict_with_text_tokens[text].count(feature)
            feature_freqs[text][feature] = presence / overall

    #      

    all_features = {}

    for feature in features:
        all_features[feature] = {}

        feature_average = 0
        for text in foldernames:
            feature_average += feature_freqs[text][feature]
        feature_average /= len(foldernames)
        all_features[feature]["Mean"] = feature_average

        # Calculate the standard deviation using the basic formula for a sample
        feature_stdev = 0
        for text in foldernames:
            diff = feature_freqs[text][feature] - all_features[feature]["Mean"]
            feature_stdev += diff*diff
        feature_stdev /= (len(foldernames) - 1)
        feature_stdev = np.sqrt(feature_stdev)
        all_features[feature]["StdDev"] = feature_stdev

    #--

    feature_zscores = {}
    for text in foldernames:
        feature_zscores[text] = {}
        for feature in features:

            # Z-score definition = (value - mean) / stddev
            feature_val = feature_freqs[text][feature]
            feature_mean = all_features[feature]["Mean"]
            feature_stdev = all_features[feature]["StdDev"]
            feature_zscores[text][feature] = ((feature_val-feature_mean) /
                                                feature_stdev)
    #
    
    users_info = pd.read_table(folder_with_pairs + '/' + dir_nr + '/users_info.tsv', sep='\t')

    result_ranking = {}
    for i in range(2):
        delta = 0
        text1 = foldernames[(i*2)]
        text2 = foldernames[(i*2+1)]
        for feature in features:
            delta += np.fabs((feature_zscores[text1][feature] -
                                feature_zscores[text2][feature]))
        delta /= len(features)
        result_ranking[text1] = delta 

    with open(burrow_file_name, 'a', newline='') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow([str(set_nr), result_ranking[foldernames[0]], result_ranking[foldernames[2]],users_info.loc[1,'int_times'],users_info.loc[0,'count_before'],
                             users_info.loc[1,'count_before'],users_info.loc[0,'count_after'],users_info.loc[1,'count_after']])
        
    #

    result_ranking = {}
    for i in range(2):
        text1 = foldernames[(i*2)]
        text2 = foldernames[(i*2+1)]
        case_values = np.array(list(feature_zscores[text1].values()))
        compare_values = np.array(list(feature_zscores[text2].values()))
        delta = np.dot(case_values, compare_values)
        delta /= np.linalg.norm(case_values)*np.linalg.norm(compare_values)
        result_ranking[text1] = np.arccos(delta)

    #

    with open(cosine_file_name, 'a', newline='') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow([str(set_nr), result_ranking[foldernames[0]], result_ranking[foldernames[2]],users_info.loc[1,'int_times'],users_info.loc[0,'count_before'],
                             users_info.loc[1,'count_before'],users_info.loc[0,'count_after'],users_info.loc[1,'count_after']])


current_directory = os.getcwd()
final_directory = os.path.join(current_directory, folder_with_pairs)
for dir_nr in os.listdir(final_directory):
    delta_for_multiple(dir_nr,set_nr)
    set_nr += 1


# In[ ]:




