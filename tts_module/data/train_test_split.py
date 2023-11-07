import os
import argparse
import random

def split_and_save(args):
    meta_all_path = args.meta_all
    meta_dir = os.path.dirname(os.path.realpath(meta_all_path))
    meta_tr_path = os.path.join(meta_dir, 'meta_train.txt')
    meta_te_path = os.path.join(meta_dir, 'meta_test.txt')
    with open(meta_all_path) as f:
        meta_all = f.readlines()
        meta_tr = []
        meta_te = []

    n_meta = len(meta_all)
    n_test = int(args.ratio_test * n_meta)
    indice_te = random.sample(range(n_meta), n_test)

    for idx, line in enumerate(meta_all):
        if idx in indice_te:
            meta_te.append(line)
        else:
            meta_tr.append(line)

    with open(meta_tr_path, 'w') as ftr:
        ftr.write(''.join(meta_tr))
    with open(meta_te_path, 'w') as fte:
        fte.write(''.join(meta_te))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split the data')
    parser.add_argument('--meta-all',  type=str, help='The meta file generated by preprocess.py', required=True)
    parser.add_argument('--ratio-test', default=0.1, type=float, help='ratio of testing examples', required=False)
    args = parser.parse_args()
    split_and_save(args)
