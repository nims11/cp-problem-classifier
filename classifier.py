from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from cnn_simple_sparse_alpha import load_data, LIMIT, SUBMISSION_SOURCE, CONCERNED_TAGS, SOURCE_THRESHOLD, get_y_vec
from collections import Counter, defaultdict
from sklearn.naive_bayes import MultinomialNB
import vectorizer
import colorsys
import numpy as np
np.random.seed(102)

def get_bytes(text):
    text = [ch for ch in vectorizer.simple_cleaner(text) if ord(ch) > 31 and ord(ch) < 128]
    return [
        bytes([ord(text[idx]), ord(text[idx-1]), ord(text[idx-2]), ord(text[idx-3])])
        for idx, _ in enumerate(text)
    ]

def get_words(text):
    text = vectorizer.alpha_cleaner(text)
    return [word.lower().strip() for line in text.split('\n') for word in line.split() if len(word) > 0]

class Classifier(object):

    def __init__(self, foo=get_words):
        self.vectorizer = TfidfVectorizer(tokenizer=foo)

    def fit(
            self,
            X,
            y,
            classifier_type=LogisticRegression,
    ):
        train_tfidf = self.vectorizer.fit_transform(X)
        classifier = classifier_type()
        classifier.fit(
            train_tfidf,
            y
        )
        self.classifier = classifier

    def predict(self, x):
        test_tfidf = self.vectorizer.transform(x)
        return self.classifier.predict(test_tfidf)

def load_data_and_labels(fname, limit=LIMIT, split=0.8):
    dataset = load_data(fname)
    x_train, y_train = [], []
    x_test, y_test = [], []
    pids = []
    problems_count = Counter([data['problem'] for data in dataset])
    problems = [k for k, v in problems_count.items() if v >= 25]
    np.random.shuffle(problems)
    np.random.shuffle(dataset)
    idx = int(0.8 * len(problems))
    train_problems, test_problems = set(problems[:idx]), set(problems[idx:])

    topic_wise_source = defaultdict(int)
    topic_wise_pid = defaultdict(set)

    for data in dataset:
        y_vec = get_y_vec(data['tags'])
        if sum(y_vec) != 1:
            continue
        if len(data['problem']) > SOURCE_THRESHOLD:
            continue

        for tag in CONCERNED_TAGS:
            if tag in data['tags']:
                topic_wise_pid[tag].add(data['problem'])
                topic_wise_source[tag]+=1

        if data['problem'] in train_problems:
            x_train.append(data['source'])
            y_train.append(CONCERNED_TAGS[y_vec.index(1)])
        else:
            x_test.append(data['source'])
            y_test.append(CONCERNED_TAGS[y_vec.index(1)])
            pids.append(data['problem'])

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), pids

import matplotlib.pyplot as plt
def get_random_color(i, n):
    hue = (360//n*i)/360.0
    sat = 0.5
    light = 0.5
    r,g,b = colorsys.hls_to_rgb(hue, light, sat)
    return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

idx = 0
def eval(classifier, X_test, Y_test, pids):
    predictions = classifier.predict(X_test)
    accuracy_sum = 0.0
    pid_count = defaultdict(lambda: defaultdict(int))
    truth = defaultdict(int)
    for y_predict, y, pid in zip(predictions, Y_test, pids):
        if y_predict == y:
            accuracy_sum += 1
        truth[pid] = y_predict
        pid_count[pid][y_predict] += 1
    
    prob_accuracy_sum = 0.0
    X = []
    Y = []
    fooo = 0
    for pid, count in sorted(pid_count.items(), key=lambda x: sum([x[1][k] for k in x[1]])):
        if truth[pid] == max(pid_count[pid].items(), key=lambda x:x[-1])[0]:
            prob_accuracy_sum += 1
        fooo += 1
        X.append(sum(pid_count[pid][k] for k in pid_count[pid]))
        Y.append(prob_accuracy_sum/fooo)

    global idx
    plt.plot(X, Y, color=get_random_color(idx, 4))
    idx += 1
    return accuracy_sum / len(Y_test), prob_accuracy_sum / len(pid_count)

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, pids = load_data_and_labels('./dataset.pickle.2')

    classifier = Classifier(foo=get_words)
    classifier.fit(X_train, Y_train)
    print('LR with alpha tokens')
    sc_accuracy, prob_accuracy = eval(classifier, X_test, Y_test, pids)
    print(sc_accuracy, prob_accuracy)

    classifier = Classifier(foo=get_bytes)
    classifier.fit(X_train, Y_train)
    print('LR with 4 byte tokens')
    sc_accuracy, prob_accuracy = eval(classifier, X_test, Y_test, pids)
    print(sc_accuracy, prob_accuracy)

    classifier = Classifier(foo=get_words)
    classifier.fit(X_train, Y_train, classifier_type=MultinomialNB)
    print('NB with alpha tokens')
    sc_accuracy, prob_accuracy = eval(classifier, X_test, Y_test, pids)
    print(sc_accuracy, prob_accuracy)

    classifier = Classifier(foo=get_bytes)
    classifier.fit(X_train, Y_train, classifier_type=MultinomialNB)
    print('NB with 4 byte tokens')
    sc_accuracy, prob_accuracy = eval(classifier, X_test, Y_test, pids)
    print(sc_accuracy, prob_accuracy)
    plt.savefig('prob.png')
