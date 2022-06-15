import pandas as pd
import json
import numpy as np
import os
import random


def conll2pandas(path: str, sep=' '):
    """Convert conll file to pandas dataframe

    Args:
        path (str): filename (eg. dataset.conll)

    Returns:
        pandas.DataFrame: pandas DataFrame with text and tags cols
    """
    with open(path, 'r', encoding='UTF8') as f:
        texts = []
        labels = []

        words = []
        tags = []
        for line in f.readlines():
            line_list = line.split(sep)
            if line_list[0] != '\n':
                words.append(line_list[0])
                tags.append(line_list[-1][:-1])
            else:
                texts.append(words.copy())
                labels.append(tags.copy())
                words.clear()
                tags.clear()

    df = pd.DataFrame()
    df['text'] = texts
    df['tags'] = labels

    return df


def conll2pandas_group_by_token(path: str, sep=' ', only_last=True):
    """Convert conll file to pandas dataframe.

    This function differs from {conll2pandas} by making
    each row of the dataframe a token, with all of it's labels.

    Args:
        path (str): filename (eg. dataset.conll)
        only_last (boolean): if set to True, only the last tag (label)
            will be put in the tags column.
    """

    with open(path, 'r', encoding='UTF8') as f:
        texts = []
        tags = []
        for line in f.readlines():
            line_list = line.replace('\n', '').replace('  ', ' ').split(sep)
            tag = line_list[-1] if only_last else line_list[1:]

            if line_list[0] != '':
                texts.append(line_list[0])
                tags.append(tag)

    return pd.DataFrame({'text': texts, 'tags': tags})


def pandas2conll(df, fname):
    """Convert pandas Dataframe to conll file

    Args:
        df (pd.DataFrame): pandas DataFrame with cols text and tags
        fname (str): filename to save eg. dataset.conll
    """
    rows = []

    for text, ent in zip(df['text'], df['tags']):
        for word, tag in zip(text, ent):
            rows.append(str(word)+' O O '+str(tag)+'\n')
        rows.append('\n')

    with open(fname, 'w', encoding="utf-8") as f:
        f.writelines(rows)


def pandas2json(df, fname: str):
    """Convert pandas to json file

    Args:
        df (pd.DataFrame): Dataframe Object
        fname (str): file name
    """

    texts = []
    for i in range(len(df)):
        text_dict = {
            "text": df['text'].iloc[i],
            "tags": df['tags'].iloc[i]
        }
        texts.append(text_dict)

    with open(fname, 'w', encoding='utf8') as file:
        for text in texts:
            json.dump(text, file, ensure_ascii=False)


def fix_seed(random_state):
    np.random.seed(random_state)
    random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    print(f'SEED {random_state} FIXED!')