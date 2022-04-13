import pandas as pd
import numpy as np
from typing import List

def filter_length_dataset(df, length_to_filter = 256):
    assert length_to_filter > 0, 'Length must be positive'
    df['quantidadeTokens'] = df['text'].apply(len)   # QUANTIDADE DE SENTENÇAS/MAX/MEDIA E MIN
    dataset_filtered = df[df['quantidadeTokens'] <= length_to_filter]  

    dataset_filtered = dataset_filtered.drop('quantidadeTokens', axis=1) # reset index
    dataset_filtered = dataset_filtered.reset_index(drop=True)
    return dataset_filtered


def filter_entities(df, minimum_entity_ratio: 0):
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

    return remove_entites(df, entities_to_remove=entities_to_remove)

def remove_entites(df, entities_to_remove: List):
    print('Entidades removidas', entities_to_remove)

    df['tags'] = df['tags'].apply(lambda tags: ['O' if tag[2:] in entities_to_remove else tag for tag in tags ] )
    
    return df



def undersampling_null_sentences(df, ratio_to_remove=0.5):
    # sentences with ALL TAGS '0'
    df['nullSentences'] = df['tags'].apply(lambda tags: all([tag == 'O' for tag in tags] ))

    df2 = df[df['nullSentences'] == True].sample(frac = ratio_to_remove, random_state = 0) 

    dataset_filtered = df[~df.index.isin(df2.index)] # todos os index que não estão nos retirados
    dataset_filtered = dataset_filtered.drop('nullSentences', axis=1) # remover a coluna criada e resetar os indexes
    
    return dataset_filtered.reset_index()

def undersampling_entity(df, undersampling_tags, ratio_to_remove=0.5):
    # sentences with at least one TAG
    df['withEntity'] = df['tags'].apply(lambda tags: any([tag[2:] in undersampling_tags for tag in tags ] ))

    df2 = df[df['withEntity'] == True].sample(frac = ratio_to_remove, random_state = 0) 

    dataset_filtered = df[~df.index.isin(df2.index)] # todos os index que não estão nos retirados
    dataset_filtered = dataset_filtered.drop('withEntity', axis=1) # remover a coluna criada e resetar os indexes
    
    return dataset_filtered.reset_index()