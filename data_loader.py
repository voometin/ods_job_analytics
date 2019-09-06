# -*- coding: utf-8 -*-

import os
import numpy as np
import json


class Dataloader():
    def __init__(self, data_folder, random_state=None):
        self.data_folder = data_folder
        self.filenames = os.listdir(data_folder)
        self.filenames_num = len(self.filenames)
        self.random_state = random_state
        
        np.random.seed(random_state)
        
    def get_random_filenames(self, n=1):
        numbers = np.random.uniform(self.filenames_num, size=(n,)).round(0).astype(int)
        for filename_num in numbers:
            yield self.filenames[filename_num]
        
    def parse_one_day(self, filename):
        js = self.load_filename(filename)
#         js = self.extract_main_keys(js)
        return js

    def load_filename(self, filename):
        with open(self.data_folder + filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def extract_main_keys(self, raw_json):
        raw_json = [{key: message.get(key, None) if key[:3]!='rea' else [(_['name'], _['count']) for _ in message.get(key, [])] for key in ['type', 'text', 'user', 'ts', 'reply_count', 'reactions']} for message in raw_json]
#         print (raw_json[0]['reactions'])
        return raw_json
