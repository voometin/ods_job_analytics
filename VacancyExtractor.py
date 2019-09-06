# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:13:18 2018

@author: Andrew
"""
import re
import pickle

from Preprocessor import Preprocessor


class VacancyExtractor():
    def __init__(self, preprocessor=None, window=70):
        self.preprocessor = preprocessor
        if self.preprocessor is None:
            self.preprocessor = Preprocessor()
            
        if self.preprocessor.pos_words == []:
            with open('./preprocessor.pkl', 'rb') as file:
                (self.preprocessor.pos_words, 
                 self.preprocessor.neg_words, 
                 self.preprocessor.ohe_dc, 
                 self.preprocessor.ohe_out_columns) = pickle.load(file)
        self.window = window
        
        self.level_re = r'\(?((?:(?:junior|jun|джун|middle|mid|мидл|senior|sr|сеньор|lead|lead(?:\s+of)?|chief(?:\s+of)?|head(?:\s+of)?|team lead|старши[йх]|старшего|ведущи[йх]|ведущего|младши[йх]|младшего|руководител[ья](?: отдела)?|глав[ау])(?:\s|\/|\+|$|\s+и|,)*)*)\)?\s*'
        self.field_re = r'(?:\W|^)?((?:(?:ai|machine learning|ml|deep learning|dl|computer vision|cv|natural language \w+|nlp|nlu|business intelligence|bi|software|research\W?|big data|python|[cс]\+\+|[cс]\#|scala|java|ios|android|devops|backend|frontend)(?:\s|\/|$)*?)*)[\)\-]?'
        self.name_vac_re = r'(?:\W|^)((?:data[\s-](?:scientist|engineer|analyst|manager)|дата[\s-](?:саентолог|сатанист|сай?ентист|инженер|аналитик)|scientist|engineer|analyst|manager|саентолог|сатанист|сай?ентист|инженер|аналитик|девелопер|программист|developer|researcher|intern|DS|дс|DE|разработчик|стажер))[sа-я]*?(?:\W|$)'

        self.reg_pat = re.compile(self.level_re + self.field_re + self.name_vac_re + self.level_re + self.field_re, flags=re.IGNORECASE|re.DOTALL)
        
        self.level_dic_normalized = {
                               'junior+': 'junior',
                               'jun+': 'junior',
                               'джун+': 'junior',
                               'jun': 'junior',
                               'джуна': 'junior',# pymorphy transofm 'джун' into 'джуна'
                               'младший': 'junior',

                               'мидл+': 'middle',
                               'middle+': 'middle',
                               'mid': 'middle',
                               'мидл': 'middle',
                               'мид': 'middle',

                               'sr+': 'senior',
                               'senior+': 'senior',
                               'сеньор+': 'senior',
                               'sr': 'senior',
                               'сеньор': 'senior',
                               'старший': 'senior',
                               'ведущий': 'senior',

                               'lead': 'head',
                               'lead of': 'head',
                               'chief': 'head',
                               'chief of': 'head',
                               'head of': 'head',
                               'team lead': 'head',
                               'руководитель': 'head',
                               'глава': 'head',
                               }
        
        self.field_dic_normalized = {
                               'machine learning': 'ml',
                               'deep learning': 'dl',
                               'computer vision': 'cv',
                               'natural language processing': 'nlp',
                               'natural language understanding':'nlu',
                               'business intelligence':'bi',
                               }
        
        self.name_vac_dic_normalized = {
                               'data scientist': 'ds',
                                'data-scientist': 'ds',
                               'scientist': 'ds',
                                'дата саентолог': 'ds',
                                'дата-саентолог': 'ds',
                                'дата сатанист': 'ds',
                                'дата-сатанист': 'ds',
                                'дата саентист': 'ds',
                                'дата-саентист': 'ds',
                                'дата сайентист': 'ds',
                                'дата-сайентист': 'ds',
                                'саентолог': 'ds',
                                'сатанист': 'ds',
                                'саентист': 'ds',
                                'сайентист': 'ds',
                                'дс': 'ds',

                                'дата инженер': 'de',
                                'дата-инженер': 'de',
                               'data engineer': 'de',
                                'де': 'de',

                                'дата аналитик': 'analyst',
                                'дата-аналитик': 'analyst',
                               'data analyst': 'analyst',
                                'аналитик': 'analyst',

                                'инженер': 'engineer',
                                'разработчик': 'engineer',
                                'девелопер': 'engineer',
                                'developer': 'engineer',
                                'программист': 'engineer',

                               'data manager': 'manager',
            
                               'стажер': 'intern',
                               }

        
    def __vacany_sort_func_importance(self, vacancy):
        if not vacancy[-1]:
            return True
        elif not re.search(r'([a-zа-я])', vacancy[-1], flags=re.IGNORECASE|re.DOTALL):
            return True

        for tkn in self.preprocessor.regexTok.tokenize(vacancy[-1].lower()):
            if self.preprocessor.morph.parse(tkn)[0].normal_form[:4] in ['vaca', 'posi', 'role', 'need', 'look', 'job', 'вака', 'назв', 'роль', 'треб', 'нужн', 'поис', 'разы', 'иска', 'необ'] or 'позици' in tkn and 'позицио' not in tkn:
                return True
        return False
    
    
    def __vacany_sort_func1(self, vacancy):
        return self.__vacany_sort_func_importance(vacancy), vacancy[0]!='', vacancy[1]!=tuple(), -vacancy[-3], -vacancy[-2]
    
    
    def parse(self, message):
        tmp = []
        for value in self.reg_pat.finditer(message):
            
            for ind, val in enumerate(value.groups()):
                if val:
                    break
            span = [value.start(ind+1), value.end(5)]
            
            value = list(map(lambda x: x.strip().lower(), value.groups()))
            
            if not value[0] and value[3]:
                value[0] = value[3]
            if not value[1] and value[4]:
                value[1] = value[4]
            value = value[:3]
            value = [
                    value[0], 
                    tuple(self.field_dic_normalized.get(x, x) for x in re.split(r'[\-\/]', value[1]) if x), 
                    self.name_vac_dic_normalized.get(self.preprocessor.morph.parse(value[2])[0].normal_form, value[2]),
                    span[0],
                    span[1],
                    re.search(r'(?:^|[\s\n\.]|[^a-z0-9а-я,])([^\.\n]{0,50})%s'%re.escape(message[span[0]:span[1]].strip()), message[max(0, span[0]-self.window):min(len(message), span[1]+self.window)], flags=re.IGNORECASE|re.DOTALL).groups()[0]
                ]

            if value[0]:
                value[0] = re.sub(r'(руков\w+)\s*отдел', '\g<1>', value[0])
                for y in re.split(r'(?<!team)(?:\s+и\s+|[,\s\/]+)(?!of)', value[0], flags=re.IGNORECASE):#.split(',и'):
                    tmp.append([self.level_dic_normalized.get(self.preprocessor.morph.parse(y)[0].normal_form, y), *value[1:]])
            else:
                tmp.append(value)

        tmp = sorted(tmp, key=self.__vacany_sort_func1, reverse=True)
        return tmp 
    
    
    def extract(self, message):
        # parse based on regex and sort extracted vacancies according to vacany_sort_func1 function
        vacancies = self.parse(message)
        # extract unique vacancies
        vacancy_unique_list = []
        for vacancy in vacancies:
            if self.__vacany_sort_func_importance(vacancy):
                if vacancy[:3] not in vacancy_unique_list and ['', tuple(), vacancy[2]] not in vacancy_unique_list: 
                    vacancy_unique_list.append(vacancy[:3])
            else:
                break
                
        return vacancies, vacancy_unique_list

# vacancies structure 
# e.g.: 
# [
#     [
#         level(jun/middle/...), 
#         [field(nlp/cv...), field, ...], 
#         vacancy_name(DS/DE/...), 
#         span_start, 
#         span_end, 
#         left_context
#     ], ...
# ]
# vacancy_unique_list - is a unique list of vacancies without span left_conext features
# e.g. ('senior', [], 'ds') and ('', [], 'ds') is the same in vacancy_unique_list
#vac_extractor = VacancyExtractor(preprocessor=preprocessor, window=70)