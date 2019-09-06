# -*- coding: utf-8 -*-

import pickle
import os

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, auc, precision_recall_curve

class Model():
    def __init__(self):
        self.model_clmns = [
            [
                  '2gte1', '1mod5==0', '1<10', 
                  '2==-1', '2mod5==0', '2<10', 
                  'curMultPeriod', 'only_digit', 
                  'all_words_score', 'short_words_score',
            ],
            [
                   '2gte1', '1mod5==0', '1<10', 
                   '2==-1', '2mod5==0', '2<10', 
                   'curMultPeriod', 'only_digit', 
                   'all_words_score', 'short_words_score', 
                   'comb0_00', 'comb0_01', 'comb0_10', 'comb0_11', 
                   'comb1_001', 'comb1_010', 'comb1_011', 'comb1_101',
#                    'comb1_000', 'comb1_001', 'comb1_010', 'comb1_101',
            ]
        ]
        self.models1_prefixes = ['_tree', '_linear']
        self.models2_prefixes = [x + '_stack' for x in self.models1_prefixes]
        self.stack_add_clmns = ['pred_proba_linear', 'pred_proba_tree',]
        self.y_col = 'target'
        self.models1 = [] # models of the first layer
        self.models2 = [] # models of the second layer - stacked models
        self.models1_names = ['tree_level_0.pkl', 'linear_level_0.pkl']
        self.models2_names = ['tree_level_1.pkl', 'linear_level_1.pkl']
        
    def __train(self, ModelClass, params, X, Y, validation=('loo', {})):
#         validation_scores = [[], [], []] # [train_acuracy_score, train_roc_auc_score, test_acuracy_score]
        validation_scores = [[], [], [], []] # [train_acuracy_score, train_roc_auc_score, train_prec_recall_auc_score, test_acuracy_score]
        if validation[0]=='loo':

            for test_index in np.arange(X.shape[0]):
                train_index = np.delete(np.arange(X.shape[0]), test_index)
                X_train, X_test = X.loc[train_index].values, X.loc[[test_index]].values
                y_train, y_test = Y.loc[train_index].values, Y.loc[[test_index]].values

                model = ModelClass(**params)
                model.fit(X_train, y_train)

                y_pred_train = model.predict_proba(X_train).T[1]
                y_pred_test = model.predict_proba(X_test).T[1]

                validation_scores[0].append(accuracy_score(y_train, np.round(y_pred_train)))
                validation_scores[1].append(roc_auc_score(y_train, y_pred_train))
                validation_scores[2].append(auc(*precision_recall_curve(y_train, y_pred_train)[:2][::-1]))
                validation_scores[3].append(accuracy_score(y_test, np.round(y_pred_test)))

            print('TRAIN: Acc: {}, Roc_auc: {}, PRC_auc: {}\nTEST: Acc: {}'.format(*np.array(validation_scores).mean(axis=1)))
        elif validation[0]=='train_test_split':
            X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, **validation[1])

            model = ModelClass(**params)
            model.fit(X_train, y_train)

            y_pred_train = model.predict_proba(X_train).T[1]
            y_pred_test = model.predict_proba(X_test).T[1]
            
            validation_scores[0].append(accuracy_score(y_train, np.round(y_pred_train)))
            validation_scores[1].append(roc_auc_score(y_train, y_pred_train))
            validation_scores[2].append(auc(*precision_recall_curve(y_train, y_pred_train)[:2][::-1]))
            validation_scores[3].append(accuracy_score(y_test, np.round(y_pred_test)))

            print('TRAIN: Acc: {}, Roc_auc: {}, PRC_auc: {}\nTEST: Acc: {}'.format(*validation_scores))

        else:#KFold
            pass

        X_train, y_train = X.values, Y.values
        model = ModelClass(**params)
        model.fit(X_train, y_train)

        return model
    
    
    def __predict(self, model, values, lcl_df1, prefix=''):
        pred_proba = np.round(model.predict_proba(values).T[1], 6)
        pred_label = np.round(pred_proba)
        lcl_df1['pred_proba'+prefix] = pred_proba
        lcl_df1['pred_label'+prefix] = pred_label
        
        
    def train(self, lcl_df):
        print('#MODEL: training start')
        # train base models
        print('Base models training')
        for index, train_params in enumerate([(DecisionTreeClassifier, {'max_depth': 4}, lcl_df[self.model_clmns[0]], lcl_df[self.y_col], ('loo', {})),
                                            (LogisticRegression, {'solver': 'liblinear'}, lcl_df[self.model_clmns[1]], lcl_df[self.y_col], ('loo', {}))]):
            self.models1.append(self.__train(*train_params))
            self.__predict(self.models1[-1], lcl_df[self.model_clmns[index]].values, lcl_df, prefix=self.models1_prefixes[index])
                           
        # train stack models - base columns + proba_predictions of previous models (linear and decision tree models)
        print('Stacked models training')
        for index, train_params in enumerate([(DecisionTreeClassifier, {'max_depth': 4}, lcl_df[self.model_clmns[0] + self.stack_add_clmns], lcl_df[self.y_col], ('loo', {})),
                                    (LogisticRegression, {'solver': 'liblinear'}, lcl_df[self.model_clmns[1] + self.stack_add_clmns], lcl_df[self.y_col], ('loo', {}))]):
            self.models2.append(self.__train(*train_params))
            self.__predict(self.models2[-1], lcl_df[self.model_clmns[index] + self.stack_add_clmns].values, lcl_df, prefix=self.models2_prefixes[index])
            
            
    def predict(self, lcl_df):
        
        for index, model in enumerate(self.models1):
            self.__predict(model, lcl_df[self.model_clmns[index]].values, lcl_df, prefix=self.models1_prefixes[index])
            
        for index, model in enumerate(self.models2):
            self.__predict(model, lcl_df[self.model_clmns[index] + self.stack_add_clmns].values, lcl_df, prefix=self.models2_prefixes[index])
            
        return lcl_df
            
    def save(self, path):
        if not os.path.exists(path):
            print (f'Couldn\'t save models. {path} doesn\'t exist')
            assert False
        # save base models
        for index, model_name in enumerate(self.models1_names):
            with open(f'{path}/{model_name}'.replace('//', '/'), 'wb') as file:
                pickle.dump(self.models1[index], file)
        # save stacked models
        for index, model_name in enumerate(self.models2_names):
            with open(f'{path}/{model_name}'.replace('//', '/'), 'wb') as file:
                pickle.dump(self.models2[index], file)
            
                           
    def load(self, path):
        if not os.path.exists(path):
            print (f'Couldn\'t load models. {path} doesn\'t exist')
            assert False
        # load base models
        for model_name in self.models1_names:
            with open(f'{path}/{model_name}'.replace('//', '/'), 'rb') as file:
                self.models1.append(pickle.load(file))
        # load stacked models
        for model_name in self.models2_names:
            with open(f'{path}/{model_name}'.replace('//', '/'), 'rb') as file:
                self.models2.append(pickle.load(file))
                           
                           
#model = Model()
#model.load('./models')
# model.train(lcl_df)
# model.save('./models')