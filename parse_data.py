import os
import sys
import pickle
from collections import defaultdict, Counter
import pprint
from multiprocessing import Pool
from lxml import html as htmldoc
from tqdm import tqdm
import vectorizer

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

def load_data(fname):
    with open(fname, 'rb') as data_file:
        return pickle.load(data_file)

DATASET = load_data('./problem_data.pickle')
DATA_DIR = './data'

def get_submission_dict():
    submission_dict = {}
    for pid, problem in DATASET.items():
        for tag in problem['tags']:
            if tag in CONCERNED_TAGS:
                for submission in problem['submissions']:
                    submission_dict[submission['id']] = {'tags': set(problem['tags']), 'problem': pid}
                break
    return submission_dict

def get_crawled_submissions():
    crawled_submissions = set()
    for root, dirs, files in os.walk(DATA_DIR):
        for name in files:
            crawled_submissions.add(int(name))
    return crawled_submissions

SUBMISSION_DICT = get_submission_dict()
CRAWLED_SUBMISSIONS = get_crawled_submissions()
del DATASET

def get_stats():
    submission_dict = SUBMISSION_DICT
    crawled_submissions = CRAWLED_SUBMISSIONS

    pp = pprint.PrettyPrinter(indent=4)
    print("Total Submissions = %d" % len(crawled_submissions))
    pp.pprint({tag: sum((1 for s in crawled_submissions if tag in submission_dict[s]['tags'])) for tag in CONCERNED_TAGS})
    pids = Counter([submission_dict[s]['problem'] for s in crawled_submissions])
    pp.pprint(Counter([count for _, count in pids.items()]))

def parse_html(file):
    with open(file) as html_file:
        html = html_file.read()
        doc = htmldoc.fromstring(html)
        return doc.xpath('//pre[contains(@class, "program-source")]//text()')[0]

def create_submission_training(sid):
    return {
        'id': sid,
        'problem': SUBMISSION_DICT[sid]['problem'],
        'tags': SUBMISSION_DICT[sid]['tags'],
        'source': vectorizer.simple_cleaner(parse_html(os.path.join(DATA_DIR, str(sid))))
    }

def create_dataset():
    out_fname = sys.argv[1]
    final_dataset = list()
    pool = Pool(8)
    for sdict in tqdm(pool.imap_unordered(create_submission_training, CRAWLED_SUBMISSIONS)):
        final_dataset.append(sdict)
    with open(out_fname, 'wb') as dump_file:
        pickle.dump(final_dataset, dump_file)

create_dataset()
# get_stats()
