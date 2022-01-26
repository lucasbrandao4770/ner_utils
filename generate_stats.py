import pandas as pd
import numpy as np


def generate_dataset_info(df, is_alldata=False, n_fold=0, train_data=True):
    assert n_fold >= 0, "n-fold cannot be negative"
    
        
    if is_alldata:
        text = [f"Processing ALL dataset statistics \n\n"]

     
    else:   
        text = [f"Processing {'train' if train_data else 'test'} dataset FOLD-{n_fold} statistics \n\n"]
    
    df['quantidadeTokens'] = df['text'].apply(len)   # QUANTIDADE DE SENTENÇAS/MAX/MEDIA E MIN
    info = df['quantidadeTokens'].describe()      
        
    # DESCRIBE INFORMATIONS
    text.append( f"{int(info['count'])} sentences\n"  )
    # COUNT TOTAL TOKENS
    text.append(  f"{str(df['quantidadeTokens'].sum())} tokens\n"   )
    text.append('\n')
    text.append( f"O tamanho médio das sentenças é: {round(info['mean'], 2)} tokens \n" )  
    text.append(  f"O tamanho máximo das sentenças é: {int(info['max'])} tokens\n"  ) 
    
    # COUNT SENTENCE WITH ALL NULL TAGS
    df['isSentenceNull'] = df['tags'].apply(lambda x: all([i=='O' for i in x]))
    text.append(  f"A quantidade de sentenças sem entidades: {str(df['isSentenceNull'].sum())}\n"  )     
    
    text.append('\n')
    # COUNT OVER 100 TOKENS PER SENTENCE
    over_100 = df[df['quantidadeTokens'] > 100].count()['text']    
    # COUNT OVER 250 TOKENS PER SENTENCE
    over_250 = df[df['quantidadeTokens'] > 250].count()['text']        
    text.append( f"O dataset possui {str(over_100)} sentenças com tamanho maior que 100 tokens\n"  )  
    text.append( f"O dataset possui {str(over_250)} sentenças com tamanho maior que 250 tokens\n"  )  
    
    
    
    text.append('\n')
    # ENTITY ANALISIS
    labels = {tag[2:]:0 for tags in df["tags"] for tag in tags if tag != "O"} # TODAS AS ENTIDADES
    entidades =  np.array([tag[2:] for tags in df["tags"] for tag in tags if tag != "O"]) # CONTAGEM DE TIPOS DE ENTIDADES
    entidades_nulas =  [tag for tags in df["tags"] for tag in tags if tag == "O"] # ENTIDADES TOTAIS NULAS

    for k in labels:   
        labels[k] = sum(entidades == k)
    
    text.append(  f"Quantidade de entidades nomeadas: {str(len(labels))}\n"  )     
    text.append(  f"Quantidade total de entidades NÃO VAZIAS: {str(len(entidades))}\n"  ) 
    text.append(  f"Quantidade total de entidades VAZIAS ('O'): {str(len(entidades_nulas))}\n"  ) 
    
    # REMOVE _ FROM LABELS AND SORTING IN DESCENDING ORDER    
    labels_sorted = {k.replace('_', ' '): v for k, v in sorted(labels.items(), key=lambda item: item[1], reverse=True)} # COLOCANDO EM ORDEM DECRESCENTE O LABELS
    text.append('\n')
    text.append("-"*15+'\n')
    #text.append("Distribuição das entidades\n")
    for k, v in labels_sorted.items():
        text.append(k+': '+ str(v) + ' tokens\n')
    text.append("-"*15+'\n\n')
    
    return text
