# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:13:18 2018

@author: Andrew
"""

import pickle
import re
from Preprocessor import Preprocessor
from Model import Model
from yargy import Parser
from yargy_parser import MONEY
from copy import deepcopy

class ForkExtractor():
    def __init__(self, preprocessor=None, model=None):
        self.preprocessor = preprocessor
        if self.preprocessor is None:
            self.preprocessor = Preprocessor()
            
        if self.preprocessor.pos_words == []:
            with open('./preprocessor.pkl', 'rb') as file:
                (self.preprocessor.pos_words, 
                 self.preprocessor.neg_words, 
                 self.preprocessor.ohe_dc, 
                 self.preprocessor.ohe_out_columns) = pickle.load(file)
            
        self.parser = Parser(MONEY)
        
        if model is None:
            self.model = Model()
            self.model.load('./models')
        else:
            self.model = model
        
    def __parse(self, last_match):
        if len(last_match)>1:# two value fork
#             print(last_match, last_match[1][0][-2:], last_match[0][0][-2:], last_match[1][0])
            if last_match[1][-2:] == last_match[0][-2:]:
                if re.search(r'^0+$', last_match[1][0]) or len(last_match[0][0])==1 and last_match[1][0][1:]=='0'*(len(last_match[1][0])-1) and len(last_match[1][0])>=3:
                    last_match[0][0] += last_match[1][0]
                    last_match[1][0] = '-1'
#             print(last_match, last_match[1][0][-2:], last_match[0][0][-2:], last_match[1][0])       
            lst = [int(last_match[0][0]), int(last_match[1][0])] + last_match[1][1:]
            last_match.pop(0)
            last_match.pop(0)
        else:# one value fork
            lst = [int(last_match[0][0]), -1] + last_match[0][1:]
            last_match.pop(0)
        return lst
            
    def parse(self, message):
        tmp = []
        last_match = []
        
        for match in self.parser.findall(message):
            match_fact = match.fact
            if match_fact:
                span = match.span
                found = re.search(r'(\S*%s\S)'%re.escape(message[span.start:span.stop]), message, flags=re.DOTALL|re.IGNORECASE)
                if found:
                    found = found.group(0)
                    if '<' in found:# any links
                        continue
                lst = list(match_fact.as_json.values()) + [span.start, span.stop]

                for y in lst[:2]:
                    if y!=-1:
                        last_match += [[x] + lst[2:] for x in re.split(r'[\s\.,]', y)]#list(map(lambda x: [x] + lst[2:], ))

                if lst[1] == -1:
                    while last_match:
                        lst = self.__parse(last_match)
                        tmp.append(lst)
                else:
                    lst = self.__parse(last_match)
                    tmp.append(lst)
    
        while last_match:
            lst = self.__parse(last_match)
            tmp.append(lst)
            
        len_tmp = len(tmp)
        if len_tmp>1:
            cnt = 0
            for ind, val in enumerate(deepcopy(tmp[:-1])):
                
                if len_tmp - 1 - cnt <= ind:
                    break
                x = tmp[ind]
                y = tmp[ind + 1]

#                 case1 example: x = [150, -1, '-', -1, '-'], y = [250, -1, '-', -1, '-']
                case1 = x[1] == -1 and y[1] == -1 

#                 ...*Условия* 1. 150 - 250р 2. Красивый и комфортный офис (Новокузнецкая/Павелецкая)...
#                 case2 example: x = [1, 150, '-', -1, '-'], y = [250, 2, '-', -1, '-']
                case2 = y[1]<y[0] and y[1]!=-1 and x[1]!=-1 and (x[0]<10 and x[1]>=10 and y[0]>10 and y[1]<=10)

                if y[-2] - 1<= x[-1] and (case1 or case2):
                    for _, val1 in enumerate(['-', -1, '-']):
                        if x[_ + 2] == val1 and y[_ + 2] != val1:
                            x[_ + 2] = y[_ + 2]

                    if case1:
                        x[1] = y[0]
                    elif case2:
                        x[0] = x[1]
                        x[1] = y[0]

                    tmp[ind] = x
                    tmp[ind][6] = y[-1]
                    tmp.remove(y)
                    cnt += 1
                    
        return tmp
    
    def parse_messages(self, messages, verbose=50):
        
        tmp = []
        if not isinstance(verbose, int):
            verbose = 0
            
        for index, message in enumerate(messages):
            
            message = message.replace('\n', ' ')#.replace('тыр', 'тыс')
            
            if verbose>0:
                if index%verbose==0:
                    print(index)
                    
            res = self.parse(message)
            tmp.append(res)
            
        return tmp
    
    def group_parsed_messages(self, results):
        # group indexes of messages by the number of the extracted forks
        
        # not found any forks
        null_indexes = []
        # found only one fork
        one_indexes = []
        # found multiple forks per message
        multi_indexes = []
        
        for index, parsed_message in enumerate(results):
            matches_number = len(parsed_message)
            if matches_number==0:
                null_indexes.append(index)
            elif matches_number>1:
                multi_indexes.append(index)
            else:
                one_indexes.append(index)
                
        return null_indexes, one_indexes, multi_indexes
    
    def extract(self, message):
        forks = self.parse(message)
        if not forks:
            return [], []
        
        lcl_df, forks = self.preprocessor.prepare_eval_data(forks, message)
        lcl_df = self.model.predict(lcl_df)
        
        
        if len(forks)>1:
            lcl_df['multi'] = 1
        else:
            lcl_df['multi'] = 0
            
        lcl_df['message_fork_found_count'] = 0
        lcl_df['pred_found_count_linear'] = lcl_df['pred_label_linear'].sum()
        lcl_df['pred_found_count_tree'] = lcl_df['pred_label_tree'].sum()
        lcl_df['res'] = 0
        lcl_df.loc[lcl_df['pred_label_linear']==lcl_df['pred_label_tree'], 'res'] = lcl_df.loc[lcl_df['pred_label_linear']==lcl_df['pred_label_tree'], 'pred_label_tree'].values
        
        lcl_df.loc[(lcl_df['res']==1) & (lcl_df['unsupervised']==0), 'res'] = 0
        lcl_df['pred_found_count_linear'] = lcl_df['res'].sum()
        lcl_df['pred_found_count_tree'] = lcl_df['pred_found_count_linear'].values
        
        if lcl_df.loc[0, 'pred_found_count_linear'] == 0:
            return forks, []

        period_dict = {'мес': 'month', 'год': 'year', 'час': 'hour', 'день': 'day', 'ч': 'hour', 'hr': 'hour', 'д': 'day'}
        lcl_df = lcl_df.replace({'5': period_dict})

        # multiplier
        lcl_df.loc[lcl_df['4']!=-1, '1'] = lcl_df.loc[lcl_df['4']!=-1, '1'].values * lcl_df.loc[lcl_df['4']!=-1, '4'].values
        lcl_df.loc[(lcl_df['4']!=-1) & (lcl_df['2']!=-1), '2'] = lcl_df.loc[(lcl_df['4']!=-1) & (lcl_df['2']!=-1), '2'].values * lcl_df.loc[(lcl_df['4']!=-1) & (lcl_df['2']!=-1), '4'].values
        lcl_df['4'] = -1
        # transform to month time period - 160 hours - 4 weeks - 20 days
        # year
        lcl_df.loc[lcl_df['5']=='year', '1'] = lcl_df.loc[lcl_df['5']=='year', '1'].values / 12.0
        lcl_df.loc[(lcl_df['5']=='year') & (lcl_df['2']!=-1), '2'] = lcl_df.loc[(lcl_df['5']=='year') & (lcl_df['2']!=-1), '2'].values / 12.0
        lcl_df.loc[lcl_df['5']=='year', '5'] = '-'
        # day
        lcl_df.loc[lcl_df['5']=='day', '1'] = lcl_df.loc[lcl_df['5']=='day', '1'].values * 20
        lcl_df.loc[(lcl_df['5']=='day') & (lcl_df['2']!=-1), '2'] = lcl_df.loc[(lcl_df['5']=='day') & (lcl_df['2']!=-1), '2'].values * 20
        lcl_df.loc[lcl_df['5']=='day', '5'] = '-'
        # hour
        lcl_df.loc[lcl_df['5']=='hour', '1'] = lcl_df.loc[lcl_df['5']=='hour', '1'].values * 160
        lcl_df.loc[(lcl_df['5']=='hour') & (lcl_df['2']!=-1), '2'] = lcl_df.loc[(lcl_df['5']=='hour') & (lcl_df['2']!=-1), '2'].values * 160
        lcl_df.loc[lcl_df['5']=='hour', '5'] = '-'
        lcl_df.loc[lcl_df['5']=='month', '5'] = '-'
        
        lcl_df.loc[(lcl_df['1'] < 1000) & (lcl_df['1']>0) & (lcl_df['3'].isin(['-', 'RUB'])), '1'] = lcl_df.loc[(lcl_df['1'] < 1000) & (lcl_df['1']>0) & (lcl_df['3'].isin(['-', 'RUB'])), '1'].values * 1000
        lcl_df.loc[(lcl_df['2'] < 1000) & (lcl_df['2']>0) & (lcl_df['3'].isin(['-', 'RUB'])), '2'] = lcl_df.loc[(lcl_df['2'] < 1000) & (lcl_df['2']>0) & (lcl_df['3'].isin(['-', 'RUB'])), '2'].values * 1000
        
        #threshold for russian and unknown currencies  - ( - 700 000] if the values of forks is out of these boundaries then the fork is not legitimable
        ru_cond = (lcl_df['3'].isin(['RUB', '-']) \
             & ((lcl_df['1']>700) & (lcl_df['1']<10000) | (lcl_df['1']>700000) | (lcl_df['1']<=12) \
        | (lcl_df['2']>700) & (lcl_df['1']<10000)  | (lcl_df['2']>700000) | (lcl_df['2']!=-1) & (lcl_df['2']<=12)
        )
        )
        eur_usd_cond = (lcl_df['3'].isin(['EUR', 'USD']) & ((lcl_df['1']<500) | (lcl_df['1']>200000) \
                        | (lcl_df['2']>=0) & ((lcl_df['2']<500) | (lcl_df['2']>200000))))

        # drop         global_df['index'].isin([179, 667, 510]) | 
        drop_index = lcl_df[ru_cond | eur_usd_cond].index
        if drop_index.shape[0]:
            lcl_df.drop(drop_index, inplace=True)
            if not lcl_df.shape[0]:
                return forks, []
        
        return forks, lcl_df.iloc[:, :3].values.tolist()

#dill._dill._reverse_typemap['ClassType'] = type
#with open('./preprocessor.pkl', 'rb') as file:
#    preprocessor = dill.load(file)
#forkExtractor = ForkExtractor()
#message = ':sberbank: :python: :tf: :book: :robot_face:\n\n*MIDDLE/SENIOR ML/AI/NLP DEVELOPER*\n\nНаша команда занимаемся задачами Искусственного Интеллекта в *Правовом департаменте Сбербанка*. Мы ищем специалистов, хорошо разбирающихся в современных технологиях машинного обучения с применением *глубоких нейросетей для задач NLP*.\n\n\n*ТЕМАТИКА*\n\nНаправление наших работ можно описать как создание ИИ-инструментов, частично или полностью заменяющих юристов при работе с внешними или внутренними контрагентами, так называемый `“Робот-Юрист”`, например:\n• `Принятие юридических заключений` по комплектам отсканированных документов, например о возможности кредитования.\n• Автоматические `ответы на юридические запросы` сотрудников банка на разнообразные темы в `диалоговом режиме`.\n• `Предсказание вероятностей положительного исхода` исков на основе накопленных исторических данных и внешних источников, являющихся `текстовыми данными`.\n• Автоматическое `формирование исковых заявлений` для подачи в суды, с учетом того, что входные данные представлены на плохо структурированном `русском языке с разнообразными ошибками`.\n• Поддержка `корпоративных и розничных клиентов` по юридическим вопросам в режиме `чат-ботов`.\n\n\n*АКТИВНОСТИ*\n\nВ процессе решения эти сверхзадачи будут распадаться на более мелкие:\n• Подготовка, очистка, расширение `датасетов`, планирование `экспериментов`.\n• `Построение моделей` с использованием оптимальных `эмбеддингов`, при необходимости — `памяти, внимания и других NLP/DL модулей`.\n• `Тренировка` моделей извлечения текстовых `сущностей`, `классификаций` документов, определения `смысловой близости`, представления `знаний`.\n• Разбор запросов, обеспечение `понимания` их `интенций`, нахождение необходимых фактов, `генерация ответов` по шаблонам и в `свободной форме`.\n• `Культурное ведение экспериментов`, проектов, отчетов, подготовка деплоев, встраивание решений в промышленную `платформу банка`.\n\n\n*ОЖИДАЕМЫЕ НАВЫКИ*\n\n• Знание `математики`, `статистики`, `алгоритмов`, `логики`, `программирования`.\n• Понимание `лингвистических` аспектов, опыт работы с пакетами типа `NLTK`, `Gensim`, `DeepPavlov`, etc.\n• Знание основ как классического `Машинного обучения`, так и `Deep Learning`.\n• Опыт построения `глубоких нейронных сетей` с использованием таких пакетов как `Tensorflow`, `Keras`, `PyTorch` etc.\n• Приветствуется как опыт `промышленной` разработки, так и участие в `стартапах`, `соревнованиях`, `исследованиях` и развитие `собственных` проектов.\n• Очевидно, не обойтись без чтения `профильной литературы`, изучения `научных статей`, постоянного `самообразования`.\n• `Дополнительные знания и опыт` (Fuzzy Logic, Formal Grammar, Reinforcement Learning, GUI, Devops, Idea generation, PM, Teaching, so on) могут повлиять на `уровень зарплаты`.\n\n\n*АТМОСФЕРА*\n\nСбербанк — крупнейшая IT-компания России, в ней работают тысячи айтишников и сотни датасайнтистов, поэтому тут идет активная жизнь, которая предполагает:\n• Участие во внутренних и внешних `комьюнити`, `семинарах`, `воркшопах`.\n• `Участие в разработке` командной и корпоративной `инфраструктуры`, например, `станций разметок`, `библиотек моделей`, `фабрик данных` и т.д.\n• Поощряется дополнительное `внешнее образование`, проводится `корпоративное обучение`.\n• Приветствуется выдвижение и проработка `инициатив`, связанных с `Искусственным Интеллектом`.\n\n\n*КОМПЕНСАЦИЯ*\n\n• Работа в Центральном аппарате Сбербанка России, м. Ленинский проспект, ул. Вавилова, д.19.\n• Социальный пакет для сотрудников (ДМС, фитнес, льготное кредитование).\n• Хороший офис, прекрасная столовая, спортзал.\n• ЗП от `180,000 до 300,000` чистыми плюс очень хорошие `годовые премии`.\n\n\nПрисылайте свое резюме мне, Роману Кошелеву, по адресу: <mailto:Koshelev.R.V@sberbank.ru|Koshelev.R.V@sberbank.ru>\nЯ буду рад увидеть в сопровождающем письме краткое изложение вашего опыта и интересов в области NLP/AI.\nДля оперативного общения используйте: Phone/WhatsApp/Telegram: <tel:+7926782-04-80|+7 926 782-04-80>'
#print(forkExtractor.extract(message))