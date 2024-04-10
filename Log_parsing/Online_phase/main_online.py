# -*- coding: utf-8 -*-
from steps.file_output_online_step import FileOutputOnlineStep
from steps.preprocess_online_step import PreprocessOnlineStep
from steps.mask_online_step import MaskOnlineStep
from steps.tokenize_online_step import TokenizeOnlineStep
from steps.dictionarize_clustering_online_step import DictClusterOnlineStep
from steps import evaluator

import os
import re
from datetime import datetime
from tqdm import tqdm
import argparse
import pandas as pd

input_dir = '../logs_Data/' # The input directory of log file
output_dir = 'LogParserResult/' # The output directory of parsing results


def load_logs(log_file, regex, headers, mode='offline', split_percentage=0.7):
    """ Function to transform log file to dataframe
        mode can be 'offline' for the first 70% of logs, or 'online' for the remaining 30%
    """
    log_messages = dict()
    linecount = 0
    total_lines = sum(1 for line in open(log_file, 'r'))
    split_line = int(total_lines * split_percentage)

    with open(log_file, 'r') as fin:
        lines = fin.readlines()
        if mode == 'offline':
            lines = lines[:split_line]
        else:  # mode == 'online'
            lines = lines[split_line:]

        for line in tqdm(lines, desc=f'load data for {mode}'):
            try:
                linecount += 1
                match = regex.search(line.strip())
                message = dict()
                for header in headers:
                    message[header] = match.group(header)
                message['LineId'] = linecount
                log_messages[linecount] = message
            except Exception as e:
                pass
    return log_messages , total_lines


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

    
ds_setting = {
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
    # 'Zookeeper': {
    #     'log_file': 'Zookeeper/Zookeeper_2k.log',
    #     'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
    #     'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?']
    # },

}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset', default='HDFS', type=str)
    parser.add_argument('--dictionary', default='../Dict.pkl', type=str)
    # parser.add_argument('--logfile', action_store=True)

    args = parser.parse_args()
    isOnline = args.online
    debug = args.debug
    dataset = args.dataset
    corpus = args.dictionary

    # setting = ds_setting[dataset]
    # ds_setting = {'OpenStack': ds_setting['OpenStack']}
    benchmark_result = []
    for dataset, setting in ds_setting.items():
        print('\n=== Evaluation on %s ==='%dataset) 
        print("------------------Online Parsing------------------------\n")
        # read file settings
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        outdir = os.path.join(output_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])
        filepath = os.path.join(indir, log_file)
        print('Parsing file: ' + filepath)

        # load templates
        # templates = pickle.load(open('templates.pkl', 'rb'))
        templates = dict()
        # load log format
        headers, regex = generate_logformat_regex(setting['log_format'])
        log_messages_offline = load_logs(filepath, regex, headers, mode='offline')
        log_messages_online, total_lines = load_logs(filepath, regex, headers, mode='online')

        # log messages is a dictionary where the key is linecount, the item is {'LineId': , header: ''}
        results = dict()
        preprocess_step = PreprocessOnlineStep(debug)
        tokenize_step = TokenizeOnlineStep(rex=setting['regex'], debug=debug)
        dict_step = DictClusterOnlineStep(corpus, debug)
        mask_step = MaskOnlineStep(dict_step, templates, results, debug)
        starttime = datetime.now()
        for lineid, log_entry in log_messages_online .items():
            if lineid in [1000, 10000, 100000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]:
                print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))
            log_entry['Message'] = log_entry['Content']
            # preprocess
            # print(log_entry['Message'])

            log_entry = preprocess_step.run(log_entry)
            # print ("***********************\n")
            # print(log_entry)

            # tokenize log content
            log_entry = tokenize_step.run(log_entry)
            # look up dictionary, return a dict: {message: log_entry['Content'], termset: validTokens, LineId: }
            termset = dict_step.run(log_entry)
            # LCS with existing templates, merging in prefix Tree
            mask_step.run(termset, log_entry)

            # print('After online parsing, templates updated: {} \n\n\n'.format(templates))
        # results = results.map(mask_layer.tagMap)
        output_file = os.path.join(outdir, log_file)
        FileOutputOnlineStep(log_messages_online, results, output_file, mask_step.cluster2Template, ['LineId'] + headers, keep_para=True).run()
        # Calculate the start line for the online phase based on the total line count and split percentage
        start_line_online = int(total_lines * 0.7) + 1
        # Read the full ground truth
        df_groundtruth_full = pd.read_csv(os.path.join(indir, log_file + '_structured.csv'))

        # Filter the ground truth for the online phase
        df_groundtruth_online = df_groundtruth_full[df_groundtruth_full['LineId'] >= start_line_online]

        # Save the filtered ground truth to a temporary file
        temp_groundtruth_path = os.path.join(indir, 'temp_groundtruth_online.csv')
        df_groundtruth_online.to_csv(temp_groundtruth_path, index=False)

        # Call the evaluate function with the adjusted ground truth for the online phase
        F1_measure, accuracy = evaluator.evaluate(
            groundtruth=temp_groundtruth_path,
            parsedresult=os.path.join(outdir, log_file + '_structured.csv')
        )
        benchmark_result.append([dataset, F1_measure, accuracy])

    print('\n=== Overall evaluation results ===')
    avg_accr = 0
    for i in range(len(benchmark_result)):
        avg_accr += benchmark_result[i][2]
    avg_accr /= len(benchmark_result)
    pd_result = pd.DataFrame(benchmark_result, columns=['dataset', 'F1_measure', 'Accuracy'])
    print(pd_result)
    print('avarage accuracy is {}'.format(avg_accr))
    pd_result.to_csv('benchmark_result.csv', index=False)


