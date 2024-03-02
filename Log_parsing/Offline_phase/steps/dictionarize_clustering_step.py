import pickle
from tqdm import tqdm
import wordninja


def hasdigit(inputString):
    return any(char.isdigit() for char in inputString)


class DictClusterStep(object):
    def __init__(self, log_messages, dictionary_file=None):
        self.log_messages = log_messages
        self.dictionary = None
        if dictionary_file:
            with open(dictionary_file, 'rb') as f:
                self.dictionary = pickle.load(f)

    def dictionaried(self):
        result = list()
        for key, value in tqdm(self.log_messages.items(), desc='dictionaried'):
            termset = list()
            for word in value['Content']:
                if hasdigit(word):
                    continue
                word = word.strip('.:*')
                if word in self.dictionary:
                    termset.append(word)
                elif all(char.isalpha() for char in word):
                    splitted_words = wordninja.split(word)
                    for sword in splitted_words:
                        if len(sword) <= 2: continue
                        termset.append(sword)
            result_dict = dict(message=value['Content'], validTokens=termset, LineId=value['LineId'])
            result.append(result_dict)
        return result

    def run(self) -> dict:
        dicted_list = self.dictionaried()
        validTokens_group = dict()
        for element in tqdm(dicted_list, desc='cluster by termset'):
            frozen_validTokens = tuple(sorted(element['validTokens']))
            if frozen_validTokens not in validTokens_group:
                validTokens_group[frozen_validTokens] = []
            validTokens_group[frozen_validTokens].append(element)
        tot = 0
        result_group = dict()
        for key in validTokens_group.keys():
            if len(key) == 0:
                for entry in validTokens_group[key]:
                    result_group[tot] = [entry]
                    tot += 1
                continue
            result_group[tot] = validTokens_group[key]
            tot += 1
        print('After Dictionarized and Clustering, total: {} bin(s)'.format(len(result_group.keys())))
        return result_group
