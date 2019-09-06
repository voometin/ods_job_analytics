# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:13:18 2018

@author: Andrew
"""

#import tensorflow as tf
#print(tf.test.gpu_device_name())

import re
import pickle

import numpy as np
import pandas as pd
import pymorphy2 as pm

from nltk.tokenize import RegexpTokenizer


class Preprocessor():
    
    def __init__(self):
        self.__percent_rule = lambda x, y: re.search(r'({}[\%\/])'.format(x), y, flags=re.DOTALL)
        self.money_default_value = [-1, '-', -1, '-']# starting from the 2nd parsed parameter
        self.morph = pm.MorphAnalyzer()
        self.regexTok = RegexpTokenizer('[\w\d\%\#]+')
        self.pos_words = []
        self.neg_words = []
        self.__y_col = 'target'
        self.dataset_clmns = ['2gte1', 'len1', '1mod5==0', '1<10', '2==-1', 'len2', '2mod5==0', '2<10', 'curMultPeriod', 'only_digit', 'context', 'local_context', 'len_tokens', 'tokens', 'all_words_score', 'short_words_score']
        self.ohe_dc = {}# ohe stands for OneHotEncoding
        self.ohe_in_columns = [['1mod5==0', '1<10'], ['2==-1', '2mod5==0', '2<10']]
#         'comb1_000', 'comb1_001', 'comb1_010', 'comb1_101'
        self.ohe_out_columns = []
        
        
    def load(self):
        with open('./preprocessor.pkl', 'rb') as file:
            (self.pos_words, 
             self.neg_words, 
             self.ohe_dc, 
             self.ohe_out_columns) = pickle.load(file)
    
    
    def __default_value_equality_count(self, x):
        cnter = 0
        
        for index in range(len(self.money_default_value)):
            if x[1 + index] != self.money_default_value[index]:
                cnter += 1
                
        return cnter
    
    
    def __unsupervised_parse_error_detect(self, cnt, tmp, raw_message):
        if cnt == 0:
            if not (tmp[0]<=10 or (tmp[0]%5!=0 or (tmp[0]>1900 and tmp[0]<2100))):
                if self.__percent_rule(tmp[0], raw_message):
                    return False
                else:
                    return True
            else:
                return False
        elif cnt==1 and tmp[1]!=-1:
            if self.__percent_rule(tmp[1], raw_message) or tmp[1]<=tmp[0] or tmp[0]<=12:
#         elif cnt==1 and tmp[1]!=-1:
#             if self.__percent_rule(tmp[1], raw_message) or tmp[1]<=tmp[0] or tmp[0]==0:
                return False
            else:
                return True
        elif tmp[1]!=-1 and tmp[1]<=tmp[0]:
            return False
        else:
            return True
    
    
    def one_index_unsupervised_parse_error_detect(self, one_indexes, results, raw_messages):
        # index - index of message
        positive = []# indexes of true positive
        negative = []# indexes of false positive

        for index in one_indexes:
            tmp = results[index][0]
            cnt = self.__default_value_equality_count(tmp)
            
            if self.__unsupervised_parse_error_detect(cnt, tmp, raw_messages[index]):
                positive.append(index)
            else:
                negative.append(index)
                
        return positive, negative
    
    
    def __money_feature_preproces(self, fork, message):
        start, end = fork[-2:]

        tmp = message[start:end]
        context = re.search(r'(?:^|[\s\n\.,:])(.{0,50}%s.{0,50})(?:$|[\s\n\.,:;])'%re.escape(tmp), message, flags=re.IGNORECASE).groups()[0]
        local_context = re.search(r'(?:^|[\s\n\.,:])?([\w\.,\/\#]*?\d+.*?)(?:$|[\.\s,])', context, flags=re.IGNORECASE).groups()[0].lower()
        if local_context[-2:]=='тр':
            fork[2:4] = 'RUB', 1000
        elif local_context[-1]=='т':
            fork[2] = 'RUB'

        only_digit = 1 if re.search(r'(^\d+(?:до|[-–—])?\d*\)?$)', local_context, flags=re.IGNORECASE) else 0

        tokens = []
        long_words_mask = []
        position = 0

        for ind, tkn in enumerate(self.regexTok.tokenize(context.lower())):
            # lemmatize
            tkn = self.morph.parse(tkn)[0].normal_form
            tokens.append(tkn)
            tkn_len = len(tkn)

            if tkn_len<4:
                long_words_mask.append(False)
            else:
                long_words_mask.append(True)
            if tkn_len<=len(local_context):
                if tkn==local_context[:tkn_len]:
                    position = ind

        position_weights_all_words = abs(np.arange(len(tokens)) - position)
        position_weights_all_words[position] = 1
        #position weight only for short words <=> words with length <=3 (or 4)
        position_weights_short_words = np.copy(position_weights_all_words)
        position_weights_short_words[long_words_mask] = 1

        position_weights_short_words = 1/position_weights_short_words
        position_weights_short_words[position] = 0

        position_weights_all_words = 1/position_weights_all_words
        position_weights_all_words[position] = 0

        return [fork + [fork[1]>=fork[0], 
                                     len(str(fork[0])), 
                                     fork[0]%5==0,
                                     fork[0]<10,
                                     fork[1]==-1,
                                     len(str(fork[1])), 
                                     fork[1]%5==0,
                                     fork[1]<10,
                                     fork[2:-2]==['-', -1, '-'],
                                     only_digit,
                                         context, local_context, 
                                     len(tokens),
                                     tokens, 
                                     position_weights_all_words, 
                                     position_weights_short_words]]

    
    def __position_weight_score(self, row):
        tokens, position_weights_all_words, position_weights_short_words = row
        values = np.array([self.pos_words.count(x) - self.neg_words.count(x) for x in tokens])
        mask = values > 0

        if mask.any():# if there is any word that appears more frequent in positive examples than in negative examples
            return [round((position_weights_all_words[mask]*np.log(values[mask])).sum()/len(tokens), 6),
                   round((position_weights_short_words[mask]*np.log(values[mask])).sum()/len(tokens), 6)]
        return [0, 0]
    
    
    def __ohe_fit_transform(self, lcl_df):
        if self.ohe_out_columns:
            self.ohe_out_columns = []

        for index, value in enumerate(self.ohe_in_columns):
            tmp = pd.get_dummies(list(map(lambda x: ''.join(map(str, x)), lcl_df[value].values)), prefix='comb%s'%index)
            lcl_df = pd.merge(lcl_df, tmp, left_index=True, right_index=True)
            self.ohe_out_columns += list(tmp.columns)
            for vals in lcl_df[value + list(tmp.columns)].drop_duplicates().values:
                self.ohe_dc[str(vals[:len(value)])] = list(vals[len(value):])

        return lcl_df

    
    def prepare_train_dataset(self, positive, negative, results, messages):
        lcl_df = []

        for cnt, index in enumerate(negative + positive):
            lcl_df += self.__money_feature_preproces(results[index][0], messages[index])
            tokens = lcl_df[-1][-3]

            if cnt>=len(negative):
                self.pos_words += list(set(tokens))
            else:
                self.neg_words += list(set(tokens))

            fork = lcl_df[-1][:7]
            results[index][0] = fork

        for index, row in enumerate(lcl_df):
#             lcl_df[index] = row[7:-2]
            lcl_df[index] = row[:-2]
            lcl_df[index][-1] = str(row[-3])
            lcl_df[index] += self.__position_weight_score(row[-3:])
            
        y_col = self.__y_col
        lcl_df = pd.DataFrame(lcl_df, columns=['1', '2', '3', '4', '5', '6', '7'] + self.dataset_clmns)
        lcl_df[y_col] = 1
        lcl_df.loc[:len(negative)-1, y_col] = 0
        lcl_df = lcl_df * 1
        try:
            lcl_df.loc[positive.index(411) + len(negative), y_col] = 0
        except:
            print('implicit target assign error')
        lcl_df.loc[lcl_df.shape[0]-1, y_col] = 0
        
        lcl_df = self.__ohe_fit_transform(lcl_df)
        
        return lcl_df
    
    
    def __ohe_transform(self, lcl_df):
        tmp = []
        for index, row in lcl_df.iterrows():
            tmp.append([])
            for clmns in self.ohe_in_columns:
#                 tmp[-1] += self.ohe_dc[str(row[clmns].values)]
                tmp[-1] += self.ohe_dc.get(str(row[clmns].values), [0, 0, 0, 0])# [0, 0, 0, 0] - for no ohe values be encoded during training step

        return pd.DataFrame(tmp, columns=self.ohe_out_columns)
    
    
    def prepare_eval_data(self, forks, message, skip_index=[]):
        lcl_df = []
        
        for index, fork in enumerate(forks):
            if index in skip_index:
                continue
            cnt = self.__default_value_equality_count(fork)
            lcl_df += self.__money_feature_preproces(fork, message)

            fork = lcl_df[-1][:7]
            forks[index] = fork

            row = lcl_df[-1]
#             lcl_df[-1] = row[7:-2]
            lcl_df[-1] = row[:-2]
            lcl_df[-1][-1] = str(row[-3])
            lcl_df[-1] += self.__position_weight_score(row[-3:])
            lcl_df[-1] += [self.__unsupervised_parse_error_detect(cnt, fork, lcl_df[-1][-6])*1]
        
        lcl_df = pd.DataFrame(lcl_df, columns=['1', '2', '3', '4', '5', '6', '7'] + self.dataset_clmns + ['unsupervised'])
#         lcl_df = pd.DataFrame(lcl_df, columns=self.dataset_clmns)
        lcl_df *= 1 #transfrom BOOL values to INT
        lcl_df = pd.merge(lcl_df, self.__ohe_transform(lcl_df), left_index=True, right_index=True)
        
        return lcl_df, forks
    
