import re

from tqdm import tqdm
import pandas as pd
import pickle
import re
class TokenizeOnlineStep():

    def __init__(self, rex=[], debug=False):
        self.rex = rex
        self.debug = debug

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, ' <*> ', line)
        return line

    def splitbychars(self, s, chars):
        l = 0
        tokens = []
        for r in range(len(s)):
            if s[r] in chars:
                tokens.append(s[l:r])
                tokens.append(s[r])
                l=r+1
        tokens.append(s[l:])
        
        tokens = list(filter(lambda x: x!='', tokens))
        # import pdb; pdb.set_trace()
        for i in range(len(tokens)):
            if all(char.isdigit() for char in tokens[i]):
                tokens[i] = '<*>'
        return tokens

    def tokenize_space(self, log_entry):
        '''
            Split string using space
        '''
        doc = self.preprocess(log_entry['Content'])
        # add other delimiters here
        terms = self.splitbychars(doc, ',;:"= []')
        log_entry['Content'] = terms
        return log_entry


    def run(self, log_entry):

        if self.debug:
            print("Starting tokenization...\n")
        results = self.tokenize_space(log_entry)
        if self.debug:
            print('Tokenization layer finished.')
        return results
