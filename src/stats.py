import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Stats:
    def __init__(self, df):
        self.df = df
        self._prepare_stats()

    def _prepare_stats(self):
        self.df["quantidadeTokens"] = self.df["text"].apply(len)

        # percorrendo todo o dataset['tags'] e verificando se todos os elementos são nulos
        self.df["isSentenceNull"] = self.df["tags"].apply(
            lambda x: all([i == "O" for i in x])
        )

        tokens_described = self.df["quantidadeTokens"].describe()

        # QUANTIDADE DE SENTENÇAS/MAX/MEDIA E MIN
        # atributes
        self.count_sentences = tokens_described["count"].astype(int)
        self.max_token = tokens_described["max"].astype(int)
        self.min_token = tokens_described["min"].astype(int)
        self.mean_token = tokens_described["mean"].round(2)
        # QUANTIDADE TOTAL DE TOKENS
        self.len_tokens = self.df["quantidadeTokens"].sum()
        # QUANTIDADE DE SENTENÇAS COMPLETAMENTE VAZIAS (SENTENÇAS NEGATIVAS)
        self.len_null_sentences = self.df["isSentenceNull"].sum()

        tags = np.array(
            [tag[2:] for tags in self.df["tags"] for tag in tags if tag[0] == "B"]
        )

        labels = dict(Counter(tags).most_common())

        self.len_labels = len(labels)
        self.len_tags = len(tags)

        # ORDEM DECRESCENTE DO LABELS E RETIRANDO _
        self.labels = {
            k.replace("_", " "): v
            for k, v in sorted(labels.items(), key=lambda item: item[1], reverse=True)
        }
        # gerando as porcentagens da quantidade das labels
        self.labels_ratio = {
            k: (v / sum(self.labels.values())) for k, v in self.labels.items()
        }
        # QUANTIDADE DE SENTENCAS ACIMA DE 256 TOKENS
        self.sentences_over_256 = self.df[self.df["quantidadeTokens"] > 256].count()[
            "text"
        ]
        # QUANTIDADE DE SENTENCAS ACIMA DE 512 TOKENS
        self.sentences_over_512 = self.df[self.df["quantidadeTokens"] > 512].count()[
            "text"
        ]

        self.negative_sentence_ratio = self.len_null_sentences / self.count_sentences

    def get_stats(self):
        # Token section
        infos = {
            "Quantidade de Sentenças": self.count_sentences,  # int
            "Quantidade de Sentenças Negativas": self.len_null_sentences,  # int
            "Quantidade de Sentenças acima de 256 tokens": self.sentences_over_256,  # int
            "Quantidade de Sentenças acima de 512 tokens": self.sentences_over_512,  # int
            "Quantidade de Tokens": self.len_tokens,  # int
            "Tamanho da maior Sentença (tokens)": self.max_token,  # int
            "Tamanho médio das Sentenças": self.mean_token,  # int
            "Quantidade de Entidades": self.len_tags,  # int
            "Quantidade de Classes": self.len_labels,  # int
            "Razão de Sentenças Negativas": self.negative_sentence_ratio,  # float
            "Labels": self.labels,  # dict
            "Labels Ratio": self.labels_ratio,  # dict
        }
        return infos


class DatasetAnalysis:
    def __init__(self, df):
        self.df = df
        self.stats = Stats(self.df).get_stats()
        self.FIG_PATH = "figs_outputs"

    def convert_stats2excel(self, save_path=""):
        token_infos = self.stats.copy()
        labels = token_infos.pop("Labels")
        labels_ratio = token_infos.pop("Labels Ratio")

        excel_sheet1 = {
            # COLUMN        # ROWS
            "Informações": token_infos,
        }

        excel_sheet2 = {
            "Quantidade de Entidades": labels,
            "Distribuição das Entidades": labels_ratio,
        }

        writer = pd.ExcelWriter(
            os.path.join(save_path, "stats_dataset.xlsx"), engine="xlsxwriter"
        )

        pd.DataFrame.from_dict(excel_sheet1, orient="columns").to_excel(
            writer, sheet_name="TokenInfo"
        )

        pd.DataFrame.from_dict(excel_sheet2, orient="columns").to_excel(
            writer, sheet_name="Entidades"
        )

        writer.close()

    def generate_dataset_info(self, is_alldata=False, n_fold=0, train_data=True) -> str:
        assert n_fold >= 0, "n-fold cannot be negative"

        if is_alldata:
            text = ["Processing ALL dataset statistics \n\n"]

        else:
            text = [
                f"""Processing {'train' if train_data else 'test'}
                dataset FOLD-{n_fold} statistics \n\n"""
            ]

        # DESCRIBE INFORMATIONS
        text.append(
            f"Razão de Sentenças Negativas {self.stats['Razão de Sentenças Negativas']}\n"
        )
        text.append(f"{int(self.stats['Quantidade de Sentenças'])} sentences\n")
        # COUNT TOTAL TOKENS
        text.append(f"{str(self.stats['Quantidade de Tokens'])} tokens\n")
        text.append("\n")
        text.append(
            f"""O tamanho médio das sentenças é:
            {(self.stats['Tamanho médio das Sentenças'])} tokens \n"""
        )
        text.append(
            f"""O tamanho máximo das sentenças é:
            {int(self.stats['Tamanho da maior Sentença (tokens)'])} tokens\n"""
        )

        # COUNT SENTENCE WITH ALL NULL TAGS
        text.append(
            f"""A quantidade de sentenças sem entidades:
            {self.stats['Quantidade de Sentenças Negativas']}\n\n"""
        )

        text.append(
            f"""O dataset possui {str(self.stats['Quantidade de Sentenças acima de 256 tokens'])}
            sentenças com tamanho maior que 256 tokens\n\n"""
        )
        text.append(
            f"""O dataset possui {str(self.stats['Quantidade de Sentenças acima de 512 tokens'])}
            sentenças com tamanho maior que 512 tokens\n\n"""
        )

        text.append(
            f"Quantidade de classes: {str(self.stats['Quantidade de Classes'])}\n"
        )
        text.append(
            f"Quantidade de entidades: {str(self.stats['Quantidade de Entidades'])}\n"
        )

        text.append("\n\n" + "-" * 15 + "\n\n")

        for k, v in self.stats["Labels"].items():
            text.append(k + ": " + str(v) + " entidades\n")
        text.append("-" * 15 + "\n\n")

        return text

    def plot_graphs(self, save_path="", verbose=False):
        if not os.path.exists(os.path.join(save_path, self.FIG_PATH)):
            os.makedirs(os.path.join(save_path, "figs_outputs"))

        sns.set()

        title = "Sentenças Positivas e Negativas"
        negative_sentences = self.stats["Razão de Sentenças Negativas"]
        positive_sentences = 1 - negative_sentences

        plt.figure(figsize=(10, 10))
        plt.pie(
            [negative_sentences, positive_sentences],
            autopct="%1.1f%%",
            labels=["Sentenças Negativas", "Sentenças Positivas"],
        )
        plt.title(title)
        plt.savefig(os.path.join(save_path, self.FIG_PATH, title))

        labels = self.stats["Labels"]
        df_tags = pd.DataFrame.from_dict(labels, columns=["Freq tags"], orient="index")
        # PORCENTAGEM
        tags_ratio = df_tags["Freq tags"] * 100 / sum(df_tags["Freq tags"])

        title = "Distribuição de Entidades"

        plt.figure(figsize=(30, 10))
        g = sns.barplot(tags_ratio, df_tags.index, palette=sns.color_palette("bright"))
        g.set_title(title)
        g.set_xlabel("Porcentagem da frequência das entidades")

        for i, v in enumerate(tags_ratio):  # escrevendo valores
            g.text(v, i, str(round(v, 2)) + "%", color="black")

        plt.savefig(os.path.join(save_path, self.FIG_PATH, title))
        if verbose:
            plt.show()
