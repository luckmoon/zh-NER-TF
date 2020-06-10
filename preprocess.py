import os
import pickle
import sys
import ast
import json
from collections import Counter

import pandas as pd

SEP = " "


def shuf_csv(src_csv, shuffed_dir):
    df = pd.read_csv(src_csv, sep="\t")
    df = df.sample(frac=1, random_state=0)
    train_df = df[:int(len(df) * 0.8)]
    test_df = df[int(len(df) * 0.8):]
    train_csv = os.path.join(shuffed_dir, "train.csv")
    test_csv = os.path.join(shuffed_dir, "test.csv")
    train_df.to_csv(train_csv, index=False, sep="\t")
    test_df.to_csv(test_csv, index=False, sep="\t")


def clean(text):
    text = text.translate(str.maketrans('', '', ' \n\t\r'))
    text = text.replace("\u3000", "").replace("\xa0", "")
    return text


def check_chonghe(l):
    l = sorted(l, key=lambda x: x[0])
    if len(l) == 1:
        return True
    for idx in range(len(l) - 1):
        if l[idx][-1] == l[idx + 1][0]:
            return True
    return False


def check_tags(title, tags):
    res = []
    for tag in tags:
        start_pos = title.find(tag)
        if start_pos == -1:
            continue
        end_pos = start_pos + len(tag)
        res.append([start_pos, end_pos])
    if not check_chonghe(res):
        return res


def parse_line(line):
    line = line.strip()
    try:
        title, tags = line.split("\t")
    except:
        return
    tags = tags.strip('[]').split(",")
    tags = [x.strip() for x in tags]
    title = clean(title)
    tags = list(map(clean, tags))
    tags = [x for x in tags if tags is not ""]
    positions = check_tags(title, tags)
    if positions:
        return title, tags, positions
    else:
        return


def gen_data(csv_file, out_file):
    with open(csv_file, 'r') as fin, open(out_file, 'w') as fout:
        next(fin)  # skip header
        for line in fin:
            tup = parse_line(line)
            if tup is not None:
                title, tags, positions = tup
                labels = ["O" for _ in range(len(title))]
                for (start, end) in positions:
                    labels[start + 1: end] = ["I"] * (end - start - 1)
                    labels[start] = "B"
                for idx in range(len(labels)):
                    fout.write("{}{}{}\n".format(title[idx], SEP, labels[idx]))
                fout.write("\n")


def save_words(train_data_file, vocab_path):
    words = []
    with open(train_data_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line is "":
                continue
            ch = line.split(SEP)[0]
            words.append(ch)
        words = Counter(words)
        words = sorted(words, key=words.get, reverse=True)

        word2id = {word: idx for idx, word in enumerate(words, 2)}
        word2id['<PAD>'] = 0,
        word2id['<UNK>'] = 1
        print(word2id)

        with open(vocab_path, 'wb') as fw:
            pickle.dump(word2id, fw)


def main():
    data_path = "./data_path2"
    csv_file = os.path.join(data_path, "original/part-00002-45492cf5-20ae-4882-ad04-dd7d7acfd8b3.csv")
    shuffed_dir = os.path.join(data_path, "shuffed")
    shuf_csv(csv_file, shuffed_dir)
    for mode in ['train', 'test']:
        shuffed_file = os.path.join(shuffed_dir, "{}.csv".format(mode))
        out_file = os.path.join(data_path, "{}_data".format(mode))
        gen_data(csv_file=shuffed_file, out_file=out_file)

    vocab_path = os.path.join(data_path, "word2id.pkl")
    save_words("./data_path2/train_data", vocab_path)


if __name__ == '__main__':
    main()
