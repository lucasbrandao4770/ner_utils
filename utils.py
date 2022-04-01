import pandas as pd
import json
import numpy as np

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
            rows.append(str(word)+' O O '+str(tag)+'\n')
        rows.append('\n')
        
    
    with open(fname, 'w', encoding="utf-8") as f:
        f.writelines(rows)

def pandas2json(df, fname: str):    
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

def filter_length_dataset(df, length_to_filter = 250):
    assert length_to_filter > 0, 'Length must be positive'
    df['quantidadeTokens'] = df['text'].apply(len)   # QUANTIDADE DE SENTENÇAS/MAX/MEDIA E MIN
    dataset_filtered = df[df['quantidadeTokens'] <= length_to_filter]  

    dataset_filtered = dataset_filtered.reset_index() # reset index
    return dataset_filtered   


def filter_entities(df, minimum_entity_ratio: 0.005):
    assert minimum_entity_ratio > 0 and minimum_entity_ratio < 1, 'Ratio must be between 0 and 1'

# TODAS AS TAGS DIFERENTES DE 'O'    
    labels = {tag[2:]:0 for tags in df["tags"] for tag in tags if tag != "O"}  # cria todas as labels com valor 0
    tags =  np.array([tag[2:] for tags in df["tags"] for tag in tags if tag != "O"]) # verifica a quantidade das labels
    #associa a label com a quantidade
    for k in labels:   
        labels[k] = sum(tags == k)

    # ORDENA CRESCENTE E DIVIDE PELO TOTAL (GERANDO PORCENTAGEM)
    total_valid_tags = sum(labels.values())
    labels = {k: v/total_valid_tags for k, v in sorted(labels.items(), key=lambda item: item[1], reverse=False)}

    entities_to_remove = []
    for k, v_ratio in labels.items():
        if v_ratio < minimum_entity_ratio:
            entities_to_remove.append(k)
        else:
            break # array está em ordem crescente, ou seja, todos os outros serão > minimo
    print('Entidades removidas', entities_to_remove)

    df['toRemoveSentence'] = df['tags'].apply(lambda tags: any( [tag[2:] in entities_to_remove for tag in tags] ))
    
    print('Quantidade de sentenças excluídas: ', (df['toRemoveSentence'] == True).sum())
    
    # filtrar o dataset com as sentenças sem as entidades selecionadas
    dataset_filtered = df[df['toRemoveSentence'] == False].drop('toRemoveSentence', axis=1) # remover a coluna auxiliar
    
    dataset_filtered = dataset_filtered.reset_index() # reset index
    return dataset_filtered 

