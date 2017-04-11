#!/usr/bin/env python3
"""
Gather submission metadata
"""
import sys
import pickle
import time
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gather_problems():
    """
    Crawler - crawl contests and problems
    """
    contests = requests.get('http://codeforces.com/api/contest.list').json()['result']
    problems = {}
    for contest in contests:
        if contest['phase'] != 'FINISHED':
            continue
        logger.info("Processing contest %d" % contest['id'])
        try:
            submissions = requests.get('http://codeforces.com/api/contest.status?contestId='+str(contest['id'])).json()['result']
        except:
            continue
        logger.info('Fetched %d submissions' % len(submissions))
        count = 0
        for submission in submissions:
            if submission['verdict'] == "OK" and 'C++' in submission['programmingLanguage']:
                count += 1
                problem = submission['problem']
                problem_id = (problem['contestId'], problem['index'])
                if problem_id not in problems:
                    problem['submissions'] = []
                    problems[problem_id] = problem
                problems[problem_id]['submissions'].append({
                    'id': submission['id'],
                    'contestId': submission['contestId'],
                    'programmingLanguage': submission['programmingLanguage'],
                })
        logger.info('Stored %d submissions' % count)
        out_file_name = sys.argv[1]
        logger.info("Dumping problem data")
        with open(out_file_name, 'wb') as dump_file:
            pickle.dump(problems, dump_file)
        time.sleep(5)

    return problems

def main():
    problems = gather_problems()


if __name__ == '__main__':
    main()
