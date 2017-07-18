import pickle
import re
import string

def hash_remover(lines):
    return [line for line in lines if (len(line) == 0 or line[0] != '#')]

indent_re = re.compile(r'\s+')
def indent_remover(lines):
    return [indent_re.sub(' ', line).strip() for line in lines]

def blank_line_remover(lines):
    return [line for line in lines if len(line) > 0]

def simple_cleaner(text):
    lines = text.split('\n')
    lines = hash_remover(lines)
    lines = indent_remover(lines)
    lines = blank_line_remover(lines)
    return '\n'.join(lines)

non_alpha_re = re.compile(r'[^a-z ]')
def non_alpha_remover(lines):
    return [non_alpha_re.sub(' ', line.lower()) for line in lines]

def alpha_cleaner(text):
    lines = text.split('\n')
    lines = non_alpha_remover(lines)
    return simple_cleaner('\n'.join(lines))

alpha_map = {}
for idx, ch in enumerate(string.ascii_lowercase):
    alpha_map[ch] = idx
alpha_map['\n'] = 26
alpha_map[' '] = 27

def alpha_ascii_vectorizer(text, limit=None):
    text = alpha_cleaner(text)
    if limit == None:
        limit = len(text)
    vector = [alpha_map[ch] for ch in text.lower()]
    while len(vector) < limit:
        vector.append(-1)
    return vector[-limit:]

def alpha_ascii_vectorizer_sparse(text, limit=None):
    if limit == None:
        limit = len(text)
    vector = alpha_ascii_vectorizer(text, limit)
    for idx, val in enumerate(vector):
        vector[idx] = [0] * 28
        if val >= 0:
            vector[idx][val] = 1
    
    return vector[-limit:]

def simple_ascii_vectorizer(text, limit=None):
    if limit == None:
        limit = len(text)
    vector = [ord(ch) - 31 for ch in text if ord(ch) >= 32 and ord(ch) < 127]
    while len(vector) < limit:
        vector.append(0)
    
    return vector[-limit:]

def simple_ascii_vectorizer_sparse(text, limit=None):
    if limit == None:
        limit = len(text)
    vector = simple_ascii_vectorizer(text, limit)
    for idx, val in enumerate(vector):
        vector[idx] = [0] * 96
        if val > 0:
            vector[idx][val] = 1
    
    return vector[-limit:]

def byte_4gram_vectorizer(text):
    pass

def variable_name_vectorizer(text):
    pass
