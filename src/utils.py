import pandas as pd
import json


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