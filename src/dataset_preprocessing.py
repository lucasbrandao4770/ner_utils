import pandas as pd
import numpy as np
from typing import List


def trucate_sentence_max_length(df, max_length=256):
    """Truncate sentences length 

    Args:
        df (pd.DataFrame): The dataframe object
        length_to_filter (int, optional): The sentence max length accepted. Defaults to 256.

    Returns:
        pd.DataFrame: The dataframe object filtered
    """
    assert max_length > 0, 'Length must be positive'

    def truncate_sentences(row, max_length=256):
        if len(row['text']) > max_length:
            row['text'] = row['text'][:max_length]
            row['tags'] = row['tags'][:max_length]

        return row

    df[['text', 'tags']] = df[['text', 'tags']].apply(
        lambda row: truncate_sentences(row, max_length=256), axis=1, result_type='expand')
    return df


def fill_O_tags(df, tags_to_remove: List):
    """Function to REPLACE a list of entities (tags) in dataFrame to 'O'
    All tags to be removed are replaced to 'O'

    Args:
        df (pd.DataFrame): The dataframe object
        tags_to_remove (List): List of entities (tags) to be removed

    Returns:
        pd.Dataframe: The dataframe object without the list of tags
    """
    print('Fill O tags', tags_to_remove)

    df['tags'] = df['tags'].apply(
        lambda tags: ['O' if tag[2:] in tags_to_remove else tag for tag in tags])

    return df


def filter_entities(df, minimum_entity_ratio: 0):
    assert minimum_entity_ratio > 0 and minimum_entity_ratio < 1, 'Ratio must be between 0 and 1'

# TODAS AS TAGS DIFERENTES DE 'O'
    labels = {tag[2:]: 0 for tags in df["tags"]
              for tag in tags if tag != "O"}  # cria todas as labels com valor 0
    # verifica a quantidade das labels
    tags = np.array([tag[2:] for tags in df["tags"]
                    for tag in tags if tag != "O"])
    # associa a label com a quantidade
    for k in labels:
        labels[k] = sum(tags == k)

    # ORDENA CRESCENTE E DIVIDE PELO TOTAL (GERANDO PORCENTAGEM)
    total_valid_tags = sum(labels.values())
    labels = {k: v/total_valid_tags for k,
              v in sorted(labels.items(), key=lambda item: item[1], reverse=False)}

    entities_to_remove = []
    for k, v_ratio in labels.items():
        if v_ratio < minimum_entity_ratio:
            entities_to_remove.append(k)
        else:
            break  # array está em ordem crescente, ou seja, todos os outros serão > minimo

    return fill_O_tags(df, entities_to_remove=entities_to_remove)


def undersampling_negative_sentences(df, ratio_to_remove=0.8):
    """Apply undersampling in sentences with full tags 'O'

    Args:
        df (pd.Dataframe): dataframe object
        ratio_to_remove (float, optional): undersampling Ratio. Defaults to 0.8.

    Returns:
        pd.dataFrame: DataFrame with undersampling
    """

    # sentences with ALL TAGS '0'
    df['nullSentences'] = df['tags'].apply(
        lambda tags: all([tag == 'O' for tag in tags]))

    df2 = df[df['nullSentences'] == True].sample(
        frac=ratio_to_remove, random_state=0)

    # todos os index que não estão nos retirados
    dataset_filtered = df[~df.index.isin(df2.index)]
    # remover a coluna criada e resetar os indexes
    dataset_filtered = dataset_filtered.drop('nullSentences', axis=1)

    return dataset_filtered.reset_index()


def undersampling_entity(df, undersampling_tags, ratio_to_remove=0.5):
    """Apply undersampling with specific tags

    Args:
        df (pd.dataFrame): Dataframe object
        undersampling_tags (List[String]): A List of Tags to apply undersampling
        ratio_to_remove (float, optional): Undersampling Ratio. Defaults to 0.5.

    Returns:
        pd.dataFrame: Dataframe object with tags undersampled
    """

    # sentences with at least one TAG
    df['withEntity'] = df['tags'].apply(lambda tags: any(
        [tag[2:] in undersampling_tags for tag in tags]))

    df2 = df[df['withEntity'] == True].sample(
        frac=ratio_to_remove, random_state=0)

    # todos os index que não estão nos retirados
    dataset_filtered = df[~df.index.isin(df2.index)]
    # remover a coluna criada e resetar os indexes
    dataset_filtered = dataset_filtered.drop('withEntity', axis=1)

    return dataset_filtered.reset_index()


def remove_jurisprudencia_sentence(df):
    # REMOVE JURISPRUDENCIA
    df['haveJurisprudencia'] = df['tags'].apply(
        lambda x: 'B-Jurisprudência' in x)
    print('SENTENÇAS COM JURISPRUDENCIA ', df['haveJurisprudencia'].sum())
    df = df[df['haveJurisprudencia'] == False][['text', 'tags']].reset_index()
    return df


def datas_change(df, datas_to_change=['Data_do_contrato', 'Data_dos_fatos']):
    # AGGREGATE datas to change with generic Datas
    df['tags'] = df['tags'].apply(
        lambda x: [tag[:2]+'Datas' if tag[2:] in datas_to_change else tag for tag in x])

    return df.reset_index()
