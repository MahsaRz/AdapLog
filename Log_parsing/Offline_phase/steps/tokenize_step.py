from tqdm import tqdm
import pickle
import re


class TokenizeStep(object):

    def __init__(self, log_messages, rex=[], dictionary_file=None):
        self.log_messages = log_messages
        self.rex = rex
        if dictionary_file:
            f = open(dictionary_file, 'rb')
            self.dictionary = pickle.load(f)
            f.close()
        else:
            self.dictionary = None

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, ' <*> ', line)
        return line

    def splitbychars(self, s, chars):
        le = 0
        tokens = []
        for r in range(len(s)):
            if s[r] in chars:
                tokens.append(s[le:r])
                tokens.append(s[r])
                le = r+1
        tokens.append(s[le:])
        tokens = list(filter(None, [token.strip() for token in tokens]))
        for i in range(len(tokens)):
            if all(char.isdigit() for char in tokens[i]):
                tokens[i] = '<*>'
        return tokens

    def tokenize_space(self):
        '''
            Split string using space
        '''
        for key, log in tqdm(self.log_messages.items(), desc='tokenization'):
            doc = self.preprocess(log['Content'])
            tokens = self.splitbychars(doc, ',;:"= ')
            log['Content'] = tokens
        return self.log_messages

    def run(self) -> list:
        results = self.tokenize_space()
        print('Tokenization step finished.')
        return results
