#!/usr/bin/env python
# coding: utf-8

# In[]:


#packages
from lxml import etree as ET 
import time
import numpy as np
import pandas as pd
from conllu import parse
from conllu import parse_incr
from conllu.models import TokenList, Token
import datetime
from dateutil.relativedelta import relativedelta
import csv
import networkx as nx
import shutil
import os
import re


# In[]:


#pre-process users and initialize sentence count
def getUsernames(xml_str):
    userTree = ET.iterparse(xml_str, tag='text', events=('end', ))
    user_dict = {}
    for _, elem in userTree:
        if elem.attrib['userid'] not in user_dict:
            user_dict[elem.attrib['userid']] = 0
        elem.clear()
        for ancestor in elem.xpath('ancestor-or-self::*'):
            while ancestor.getprevious() is not None:
                del ancestor.getparent()[0]
    del userTree
    return user_dict


# In[]:


#The folder to extract too and subforum to extract from. The XML-files are from Språkbanken Text
folder_str = "Across-folder/"
xml_str = 'flashback-ekonomi.xml'
user_dict = getUsernames(xml_str)
xml_str = 'flashback-mat.xml'
user_dict1 = getUsernames(xml_str)


# In[]:


#create text files with text and information in CONLLU file format while also counting the number of tokens
def extractXML(xml_str,thread_dict,user_dict):
    xmlTree = ET.iterparse(xml_str, tag=('text','w'), events=('end', ))
    date_of_post = '' #either '%Y-%m-%d %H:%M:%S' or '%Y-%m-%d %H:%M' later on
    sentence_CON = TokenList([])
    duplicate_set = set([]) #Språkbanken Text suffered from duplicates of texts extracted from Flashback
    token_str = ''
    for _, branch in xmlTree:
        if branch.tag == 'w':
            token_str += branch.text
            sentence_CON.append({"id": int(branch.attrib['ref']), "form": branch.text, "lemma": xml_str,
                                 "upos": branch.attrib['pos'], "xpos": branch.attrib['msd'], "date": ""}) #more information available this was decided to be the most useful
        
        elif branch.tag == 'text':
            sentence_id = branch.attrib['id'] + token_str
            if sentence_id not in duplicate_set:
                duplicate_set.add(sentence_id)
                date_of_post = branch.attrib['date']
                if branch.attrib['userid'] in user_dict:
                    user_dict[branch.attrib['userid']] += 1 
                    sentence_CON[0]["date"] = date_of_post
                    with open(folder_str + branch.attrib['userid'] + ".CONLLU", 'a', encoding='utf-8') as out_file:
                        try:
                            out_file.writelines([sentence_CON.serialize()]) # + "\n" for sentence in a
                        except:
                            pass
                        out_file.close()
                    
                thread_id = branch.getparent().attrib['id']
                if thread_id not in thread_dict:
                    thread_dict[thread_id] = set([(branch.attrib['userid'], date_of_post)])
                else:
                    thread_dict[thread_id].add((branch.attrib['userid'], date_of_post))
            
            sentence_CON = TokenList([])
            token_str = ''
            
        branch.clear()
        for ancestor in branch.xpath('ancestor-or-self::*'):
            while ancestor.getprevious() is not None:
                del ancestor.getparent()[0]
    del xmlTree
    


# In[]:


#thread_dict stores all posts by userid and date in a thread 
xml_str = 'flashback-ekonomi.xml'
thread_dict = {}
extractXML(xml_str,thread_dict,user_dict) #start extraction


# In[]:


for idnr in list(user_dict.keys()):
    if user_dict[idnr] < 400:
        del user_dict[idnr]
        try:
            os.remove(folder_str + idnr + ".CONLLU")
        except:
            continue


# In[]:


for thread_id in thread_dict:
    thread_dict[thread_id] = np.array(sorted(np.array(list(thread_dict[thread_id])), 
                    key= lambda x: datetime.datetime.strptime(x[1], '%Y-%m-%d %H:%M')))


# In[]:


#Here are interactions for every pair counted. 
dynamic_dict = {}

def add_to_dicts(user_pair,check_id):
    dynamic_dict[user_pair]['dates'].add(check_id[1])
    times = dynamic_dict[user_pair]['times']
    times += 1
    dynamic_dict[user_pair]['times'] = times


for thread_id in thread_dict:
    four_names = []
    for user_id in thread_dict[thread_id]:
        four_names.append(user_id)
        if len(four_names)>= 4:
            name = four_names.pop(0)
            if name[0] in user_dict:
                for check_id in four_names:    
                    if check_id[0] in user_dict and (name[0] != check_id[0]):
                        pair_str1 = name[0] + '&' + check_id[0]
                        pair_str2 = check_id[0] + '&' + name[0]
                        if pair_str1 in dynamic_dict:
                            add_to_dicts(pair_str1,check_id)
                        elif pair_str2 in dynamic_dict:
                            add_to_dicts(pair_str2,check_id)     
                        else:
                            dynamic_dict[pair_str1] = {}
                            dynamic_dict[pair_str1]['dates'] = set([check_id[1]])
                            dynamic_dict[pair_str1]['times'] = 1

    if len(four_names)-1 > 0:
        for i in range(len(four_names)-1):
            name = four_names.pop(0)
            if name[0] in user_dict:
                for check_id in four_names:    
                    if (check_id[0] in user_dict) and (name[0] != check_id[0]):
                        pair_str1 = name[0] + '&' + check_id[0]
                        pair_str2 = check_id[0] + '&' + name[0]
                        if pair_str1 in dynamic_dict:
                            add_to_dicts(pair_str1,check_id)
                        elif pair_str2 in dynamic_dict:
                            add_to_dicts(pair_str2,check_id)        
                        else:
                            dynamic_dict[pair_str1] = {}
                            dynamic_dict[pair_str1]['dates'] = set([check_id[1]])
                            dynamic_dict[pair_str1]['times'] = 1


# In[]:


#DYNAMIC CONVERGENCE. This is the normal conversion with all text included for every pair
current_directory = os.getcwd()
folder_to_set = '' #example: pair_folder/set
from_folder = folder_str
interaction_threshold = 5 #Deside an interaction threshold 
for i, user_pair in enumerate(dynamic_dict):
    
    times = dynamic_dict[user_pair]['times']
    
    if times < interaction_threshold:
        continue
    
    interaction_dates = sorted(list(dynamic_dict[user_pair]['dates']), 
                    key= lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M')) #'%Y-%m-%d %H:%M:%S:' familjeliv
    first_date = interaction_dates[0]
    threshold_date = datetime.datetime.strptime(first_date, '%Y-%m-%d %H:%M')+ relativedelta(months=1)
            
    final_directory = os.path.join(current_directory, folder_to_set+str(i))
    os.mkdir(final_directory)
        
    user_list = user_pair.split('&')
    sen_counts = [0,0,0,0] #this list will contain the total sentence count before and after interacting
    listdates = [first_date,first_date,threshold_date,threshold_date] #this list will contain the very first and last post made by a user
    
    for j in range(2):
        data_file = open(from_folder + user_list[j] + ".CONLLU", "r", encoding="utf-8")
        
        for tokenlist in parse_incr(data_file):
            sen_date = tokenlist[0]["xpos"] 
            if (datetime.datetime.strptime(sen_date, '%Y-%m-%d %H:%M') < 
            datetime.datetime.strptime(first_date, '%Y-%m-%d %H:%M')):
                
                with open(folder_to_set + str(i) + '/before' + str(j) + ".CONLLU", 'a', 
                          encoding='utf-8') as out_file:
                    try:
                        out_file.writelines([tokenlist.serialize()]) 
                    except:
                        pass
                    out_file.close()
                if (datetime.datetime.strptime(sen_date, '%Y-%m-%d %H:%M') < 
            datetime.datetime.strptime(listdates[j], '%Y-%m-%d %H:%M')):
                    listdates[j] = sen_date
                sen_counts[j] += 1
            elif (datetime.datetime.strptime(sen_date, '%Y-%m-%d %H:%M') >
            datetime.datetime.strptime(threshold_date, '%Y-%m-%d %H:%M')):
                
                with open(folder_to_set + str(i) + '/after' + str(j) + ".CONLLU", 'a', 
                          encoding='utf-8') as out_file:
                    try:
                        out_file.writelines([tokenlist.serialize()]) 
                    except:
                        pass
                    out_file.close()
                if (datetime.datetime.strptime(sen_date, '%Y-%m-%d %H:%M') >
            datetime.datetime.strptime(listdates[j+2], '%Y-%m-%d %H:%M')):
                    listdates[j+2] = sen_date
                sen_counts[j+2] += 1    
                    
        data_file.close()
        
    if all(i >= 400 for i in sen_counts):
        with open(folder_to_set + str(i) + "/users_info.tsv", 'w', newline='') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['first_date', 'last_date','interaction_date','threshold_date','count_before','count_after'])
            tsv_writer.writerow([listdates[0], listdates[2],first_date,threshold_date,sen_counts[0],sen_counts[2]])
            tsv_writer.writerow([listdates[1], listdates[3],first_date,threshold_date,sen_counts[1],sen_counts[3]])
            out_file.close()
    else:
        shutil.rmtree(final_directory, ignore_errors=True)

