# -*- coding: utf-8 -*-
# !/usr/bin/env python
from steps.preprocess_step import PreprocessStepLayer
from steps.tokenize_step import TokenizeStep
from steps.dictionarize_clustering_step import DictClusterStep
from steps.LCS_merging_step import MaskStep
from steps.file_output_step import FileOutputStep
from steps import evaluator
import os
import re
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import argparse

input_dir = '../logs_Data/'  # The input directory of log file
output_dir = 'LogParserResult/'  # The output directory of parsing results


def load_logs(log_file, regex, headers):
    """ Function to transform log file to dataframe
    """
    log_messages = dict()
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in tqdm(fin.readlines(), desc='load data'):
            try:
                linecount += 1
                match = regex.search(line.strip())
                message = dict()
                for header in headers:
                    message[header] = match.group(header)
                message['LineId'] = linecount
                log_messages[linecount] = message
            except Exception:
                pass
    return log_messages


def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex


benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
    },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+']
    },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+']
    },


    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\s\d+\s']
    },

}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dictionary', default='../Dict.pkl', type=str)
    args = parser.parse_args()
    corpus = args.dictionary

    benchmark_result = []
    for dataset, setting in benchmark_settings.items():
        print('\n=== Evaluation on %s ===' % dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        outdir = os.path.join(output_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])

        filepath = os.path.join(indir, log_file)
        print('Parsing file: ' + filepath)
        starttime = datetime.now()
        headers, regex = generate_logformat_regex(setting['log_format'])
        log_messages = load_logs(filepath, regex, headers)
        # preprocess step
        log_messages = PreprocessStepLayer(log_messages).run()
        # tokenize step
        log_messages = TokenizeStep(log_messages, rex=setting['regex']).run()
        # dictionarize step and cluster by termset
        dict_group_result = DictClusterStep(log_messages, corpus).run()
        # LCS and prefix tree steps
        results, templates = MaskStep(dict_group_result).run()
        output_file = os.path.join(outdir, log_file)
        # output parsing results
        FileOutputStep(log_messages, output_file, templates, ['LineId'] + headers).run()
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))
        F1_measure, accuracy = evaluator.evaluate(
            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
            parsedresult=os.path.join(outdir, log_file + '_structured.csv')
        )
        benchmark_result.append([dataset, F1_measure, accuracy])

    print('\n=== Overall evaluation results ===')
    avg_acc = 0
    for i in range(len(benchmark_result)):
        avg_acc += benchmark_result[i][2]
    avg_acc /= len(benchmark_result)
    pd_result = pd.DataFrame(benchmark_result, columns=['dataset', 'F1_measure', 'Accuracy'])
    print(pd_result)
    print('Average accuracy is {}'.format(avg_acc))
    pd_result.to_csv('benchmark_result.csv', index=False)
