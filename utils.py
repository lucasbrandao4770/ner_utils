import pandas as pd


def conll2pandas(path: str):
    """Convert conll file to pandas dataframe

    Args:
        path (str): filename (eg. dataset.conll)

    Returns:
        pandas.DataFrame: pandas DataFrame with text and tags cols
    """
    f = open(path, 'r', encoding='UTF8')

    texts = []
    labels = []

    words = []
    tags = []
    for line in f.readlines():  
        line_list = line.split(' ')
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
            rows.append(str(word)+'\t'+str(tag)+'\tO\tO\n')
        rows.append('\n')
        
    
    f = open(fname, 'w', encoding="utf-8")
    f.writelines(rows)





