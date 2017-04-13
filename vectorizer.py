import pickle
import re

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

def simple_ascii_vectorizer(text, limit=None):
    if limit == None:
        limit = len(text)
    vector = [ord(ch) for ch in text]
    while len(vector) < limit:
        vector.append(0)
    
    # Try fetching last n bytes instead
    return vector[-limit:]

def byte_4gram_vectorizer(text):
    pass

def variable_name_vectorizer(text):
    pass
