import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import conll2pandas


class Stats:
    def __init__(self, df):
        self.df = df       
        self._prepare_stats()
     
    def _prepare_stats(self):
        self.df['quantidadeTokens'] = self.df['text'].apply(len)  
        
        # percorrendo todo o dataset['tags'] e verificando se todos os elementos são nulos
        self.df['isSentenceNull'] = self.df['tags'].apply(lambda x: all([i=='O' for i in x])) 

        tokens_described = self.df['quantidadeTokens'].describe()
   

        # QUANTIDADE DE SENTENÇAS/MAX/MEDIA E MIN
        # atributes
        self.count_sentences = tokens_described['count']
        self.max_token = tokens_described['max']
        self.min_token = tokens_described['min']
        self.mean_token = tokens_described['mean'].round(2)        
        # QUANTIDADE TOTAL DE TOKENS
        self.len_tokens = self.df['quantidadeTokens'].sum()
        # QUANTIDADE DE SENTENÇAS COMPLETAMENTE VAZIAS
        self.len_null_sentences = self.df['isSentenceNull'].sum() 

        # TODAS AS TAGS DIFERENTES DE 'O'
        labels = {tag[2:]:0 for tags in self.df["tags"] for tag in tags if tag != "O"} 
        tags =  np.array([tag[2:] for tags in self.df["tags"] for tag in tags if tag != "O"]) 

        for k in labels:   
            labels[k] = sum(tags == k)

        self.len_labels = len(labels)
        self.len_tags = len(tags)
        tags_null =  [tag for tags in self.df["tags"] for tag in tags if tag == "O"] # tags TOTAIS NULAS
        self.len_null_tags = len(tags_null)

        # ORDEM DECRESCENTE DO LABELS E RETIRANDO _ 
        self.labels = {k.replace('_', ' '): v for k, v in sorted(labels.items(), key=lambda item: item[1], reverse=True)}

        # QUANTIDADE DE SENTENCAS ACIMA DE 250 TOKENS
        self.sentences_over_250 = self.df[self.df['quantidadeTokens'] > 250].count()['text']        

        
    def get_stats(self):
        # Token section
        infos = {
            'Sentences Count': self.count_sentences, #int
            'Sentences Null Length': self.len_null_sentences, #int
            'Sentences Over 250 tokens': self.sentences_over_250, #int
            'Tokens Count': self.len_tokens, #int
            'Tokens Max Length': self.max_token, #int
            'Tokens Mean Length': self.mean_token,  #int
            'Tags Length': self.len_tags, #int
            'Tags Null Length': self.len_null_tags, #int 
            'Labels Length': self.len_labels, #int
            'Labels': self.labels, #dict               
        }
        return infos
        # Labels Sections
        # labels = {'Labels': self.labels} 

        # return {                           
        #         'Tokens': token_infos, #dict
        #         'Labels': labels # dict                           
        #     }


class DatasetAnalysis:
    def __init__(self, path, df=None):
        self.df = conll2pandas(path) #if df!=None else df
        self.stats = Stats(self.df).get_stats()
        self.time_stamp = self._get_df_timestamp(path)
        self.FIG_PATH = 'figs_outputs'

    def _get_df_timestamp(self, path, data_format = '%m-%Y'):
        assert data_format in ['%Y-%m-%d', '%m-%Y', '%d-%m-%Y'], 'Invalid data format'
        ti_c = os.path.getctime(path) 
        time_created = time.ctime(ti_c) 
        t_obj = time.strptime(time_created) 
        return time.strftime(data_format, t_obj) # OR "%d-%m-%Y" OR "%Y-%m-%d"
        
    def convert_stats2excel(self, save_path = ''):         
        token_infos = self.stats.copy()
        labels = token_infos.pop('Labels')
       
        excel_sheet1 = {                  
            # COLUMN        # ROWS
            'Tokens infos': token_infos,                                           
        }
        
        excel_sheet2 = {
            'Quantidade de Labels': labels
        }
        writer = pd.ExcelWriter(os.path.join(save_path, f'analysis_{self.time_stamp}.xlsx'), engine='xlsxwriter')

        pd.DataFrame.from_dict(excel_sheet1, orient='columns')\
            .to_excel(writer, sheet_name='TokenInfo')
        
        pd.DataFrame.from_dict(excel_sheet2, orient='columns')\
            .to_excel(writer, sheet_name = 'Entidades')

        writer.close()

    def generate_dataset_info(self, df, is_alldata=False, n_fold=0, train_data=True) -> str:
        assert n_fold >= 0, "n-fold cannot be negative"

        if is_alldata:
            text = [f"Processing ALL dataset statistics \n\n"]
        
        else:   
            text = [f"Processing {'train' if train_data else 'test'} dataset FOLD-{n_fold} statistics \n\n"]
        
                  
        # DESCRIBE INFORMATIONS
        text.append( f"{int(self.stats['Sentences Count'])} sentences\n"  )
        # COUNT TOTAL TOKENS
        text.append(  f"{str(self.stats['Tokens Count'])} tokens\n"   )
        text.append('\n')
        text.append( f"O tamanho médio das sentenças é: {int(self.stats['Tokens Mean Length'])} tokens \n" )  
        text.append(  f"O tamanho máximo das sentenças é: {int(self.stats['Tokens Max Length'])} tokens\n"  ) 
        
        # COUNT SENTENCE WITH ALL NULL TAGS
        text.append(  f"A quantidade de sentenças sem entidades: {self.stats['Sentences Null Length']}\n\n"  )     
       
        text.append( f"O dataset possui {str(self.stats['Sentences Over 250 tokens'])} sentenças com tamanho maior que 250 tokens\n\n"  )  
                    
        text.append(  f"Quantidade de classes: {str(self.stats['Labels Length'])}\n"  )     
        text.append(  f"Quantidade de tags NÃO VAZIAS: {str(self.stats['Tags Length'])}\n"  ) 
        text.append(  f"Quantidade de tags VAZIAS ('O'): {str(self.stats['Tags Null Length'])}\n"  ) 
        
        text.append('\n\n' + "-"*15)
       
        for k, v in self.stats['Labels'].items():
            text.append(k+': '+ str(v) + ' tokens\n')
        text.append("-"*15+'\n\n')
        
        return text
        

    def plot_graphs(self, save_path = ''):
        if not os.path.exists(os.path.join(save_path, self.FIG_PATH)):
            os.makedirs(os.path.join(save_path, 'figs_outputs'))

        sns.set()
        title=f"Entidades Válidas x Nulas - COREJUR {self.time_stamp}"
        plt.figure(figsize=(10,10))
        len_tags_null = self.stats['Tags Null Length']
        len_tags = self.stats['Tags Length']
        plt.pie([len_tags_null, len_tags], autopct='%1.1f%%', labels=['Entidades Nulas', 'Entidades Válidas'])
        plt.title(title)        
        plt.savefig(os.path.join(save_path, self.FIG_PATH, title))

        labels = self.stats['Labels']
        df_tags = pd.DataFrame.from_dict(labels, columns=['Freq tags'], orient='index')
        # PORCENTAGEM
        tags_ratio = df_tags['Freq tags'] * 100 / sum(df_tags['Freq tags'])  

        title = f'Distribuição de Entidades {self.time_stamp}'
        plt.figure(figsize=(30,10))
       
        g = sns.barplot(tags_ratio, df_tags.index, palette=sns.color_palette('bright'))
        g.set_title(title)
        g.set_xlabel('Porcentagem da frequência das entidades')

        for i, v in enumerate(tags_ratio): # escrevendo valores
            g.text(v, i, str(round(v, 2)) + '%', color='black')

        plt.savefig(os.path.join(save_path, self.FIG_PATH, title))
        plt.show()  


