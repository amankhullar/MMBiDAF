import os
import re
import sys

import numpy as np

from rouge import Rouge
from nltk import PorterStemmer

stemmer = PorterStemmer()

def get_num(str):
    return int(re.search(r'\d+', str).group())

def prepare(gt, res):
	clean_gt = [" ".join([stemmer.stem(i) for i in line.split()]) for line in gt]
	clean_res = [" ".join([stemmer.stem(i) for i in line.split()]) for line in res]
	return clean_gt, clean_res

def get_rouge(clean_gt, clean_res):
    rouge = Rouge()
    scores = rouge.get_scores(clean_res, clean_gt, avg=True)
    return scores

def calculate_rouge(gt_file, res_file):
    try:
        with open(gt_file, 'r') as f:
            gt_data = f.readlines()
        with open(res_file, 'r') as f:
            res_data = f.readlines()
    except Exception as e:
        print("Cannot open files with error : "+ e)
    else:
        min_len = min(len(gt_data), len(res_data))
        gt_data = gt_data[:min_len]
        res_data = res_data[:min_len]
        clean_gt, clean_res = prepare(gt_data, res_data)
        score = get_rouge(clean_gt, clean_res)
    return score

def main(courses_dir, res_dir):
    courses_dirlist = []
    for fname in os.listdir(courses_dir):
        if os.path.isdir(os.path.join(courses_dir, fname)):
            courses_dirlist.append(fname)

    res_dirlist= []
    for fname in os.listdir(res_dir):
        if os.path.isdir(os.path.join(res_dir, fname)):
            res_dirlist.append(fname)

    assert len(res_dirlist) == len(courses_dirlist), "Unequal ground truth and generated summary dir length"

    num_files = 0
    total_score_r1p = 0
    total_score_r1r = 0
    total_score_r1f = 0
    total_score_r2p = 0
    total_score_r2r = 0
    total_score_r2f = 0
    total_score_rl1 = 0
    total_score_rl2 = 0
    total_score_rlf = 0

    for course_num in sorted(courses_dirlist, key=int):
        gt_path = os.path.join(courses_dir, course_num, 'ground-truth')
        res_path = os.path.join(res_dir, course_num)
        for gt, res in zip(sorted(os.listdir(gt_path), key=get_num), sorted(os.listdir(res_path), key=get_num)):
            if '.txt' in gt and '.txt' in res:
                num_files += 1
                gt_file = os.path.join(gt_path, gt)
                res_file = os.path.join(res_path, res)
                total_score = calculate_rouge(gt_file, res_file)
                total_score_r1p += total_score['rouge-1']['p']
                total_score_r1r += total_score['rouge-1']['r']
                total_score_r1f += total_score['rouge-1']['f']
                total_score_r2p += total_score['rouge-2']['p']
                total_score_r2r += total_score['rouge-2']['r']
                total_score_r2f += total_score['rouge-2']['f']
                total_score_rl1 += total_score['rouge-l']['p']
                total_score_rl2 += total_score['rouge-l']['r']
                total_score_rlf += total_score['rouge-l']['f']

    total_score_r1p /= num_files
    total_score_r1r /= num_files
    total_score_r1f /= num_files
    total_score_r2p /= num_files
    total_score_r2r /= num_files
    total_score_r2f /= num_files
    total_score_rl1 /= num_files
    total_score_rl2 /= num_files
    total_score_rlf /= num_files
    print("Average rouge score over all the files is : ", total_score_r1p, total_score_r1r, total_score_r1f, total_score_r2p, total_score_r2r, total_score_r2f, total_score_rlp, total_score_rlr, total_score_rlf)


if __name__ == "__main__":
    courses_dir = ""
    res_dir = ""
    main(courses_dir, res_dir)
