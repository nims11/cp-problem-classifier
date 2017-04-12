import os
import random
import pickle
import time
import lxml
import subprocess

CONCERNED_TAGS = [
    'binary search',
    'bitmasks',
    'dp',
    'dsu',
    'flows',
    'geometry',
    'graphs',
    'shortest paths',
    'trees'
]

PROBLEMS_PER_TAG = 75

def load_data(fname):
    with open(fname, 'rb') as data_file:
        return pickle.load(data_file)

DATASET = load_data('./problem_data.pickle')

def get_concerned_problems():
    concerned_problems = set()
    for tag in CONCERNED_TAGS:
        count = 0
        for pid, problem in sorted(DATASET.items(), key=lambda x: -x[0][0]):
            if tag in problem['tags']:
                concerned_problems.add(pid)
                count += 1
            if count == PROBLEMS_PER_TAG:
                break
    return concerned_problems

CONCERNED_PROBLEMS = get_concerned_problems()
DATA_DIR = './data'

def get_concerned_submissions():
    submission_pool = {}
    for pid in CONCERNED_PROBLEMS:
        submission_pool[pid] = [sub['id'] for sub in DATASET[pid]['submissions']]
        random.shuffle((submission_pool[pid]))

    while len(submission_pool) > 0:
        for pid, lst in list(submission_pool.items()):
            fname = os.path.join(DATA_DIR, '%d' % lst[0])
            if not os.path.exists(fname):
                url = 'http://codeforces.com/contest/%d/submission/%d' % (pid[0], lst[0])
                subprocess.call('wget %s --directory-prefix=%s' % (url, DATA_DIR), shell=True)
                time.sleep(1)
            submission_pool[pid] = lst[1:]

get_concerned_submissions()
