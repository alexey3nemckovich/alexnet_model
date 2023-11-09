from .string_utils import normalize_string


def read_pairs(file_path):
    lines = open(file_path, encoding='utf-8').read().strip().split('\n')
    pairs = [tuple([normalize_string(s) for s in l.split('\t')[:2]]) for l in lines]
    
    return pairs
