"""Small tool to parse the log the quickly find the best prec@1
"""
import sys
import os
import argparse


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_fname', type=str)

    args = parser.parse_args()
    assert os.path.isfile(args.log_fname)
    return args


def parse_line(line):
    tokens = line.split(',')
    loss = float(tokens[0].strip().split(' ')[1])
    prec1 = float(tokens[1].strip().split(' ')[1])
    prec5 = float(tokens[2].strip().split(' ')[1])
    return loss, prec1, prec5


def main(args):
    """Main function"""
    content = open(args.log_fname, 'r').read().splitlines()

    best_prec1 = 0
    best_idx = 0

    for i, line in enumerate(content):
        if 'Testing results:' in line:
            loss, prec1, prec5 = parse_line(content[i+1])
            if prec1 > best_prec1:
                best_prec1 = prec1
                best_idx = i

    print(content[best_idx+1])
    print(content[best_idx+2])
    print(content[best_idx+3])
    if (best_idx+4 < len(content)) and (content[best_idx+4].startswith('  Belief Loss')):
        print(content[best_idx+4])
    return 0


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
