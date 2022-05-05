redator = ['B-Valor_dano_moral',
'B-Data_do_contrato',
'B-CNPJ_do_réu',
'B-CPF_do_réu']

auxiliar = ['B-Valor_danos_materiais/restituição_em_dobro',
'B-Valor_da_causa',
'B-Valor_da_multa_–_Tutela_provisória',
'B-Valores',
'B-Data_da_petição',
'B-Data_dos_fatos',
'B-Datas',
'B-CNPJ_do_autor',
'B-CNPJ',
'B-CPF_do_autor',
'B-CPF']


def find_indexes_from_label(dataset, label, tags_column=1):
  """Encontra todas as ocorrências de uma label
  em um dataset.
  """
  relevant_indexes = []
  for row in range(dataset.shape[0]):
    # encontra as tags da linha
    tags = dataset.iloc[row][tags_column]

    # checa se 'label' é uma das tags
    if label in tags:
      relevant_indexes.append(row)

  return relevant_indexes




def balance_entity(destination, source, qtd, entity, normalize_qtd=True):
  """Transfere dado número de ocorrências de uma entidade
  do dataset de origem (source) para o dataset de destino (destination).

  Por padrão, o valor 'qtd' será normalizado para um valor positivo para
  garantir que o número de entidades transferida permaneça o mesmo.

  Passar um valor negativo de 'qtd' e inicializar a variável 'normalize_qtd'
  como 'False' fará com que sejam transferidas 'n - qtd' ocorrências para o
  dataset de destino (onde 'n' é o número total de ocorrências).

  Exemplo: caso existam 7 ocorrências de 'entity' em 'source':
    qtd = 2, normalize_qtd = True -> 2 entidades transferidas
    qtd = 2, normalize_qtd = False -> 2 entidades transferidas
    qtd = -2, normalize_qtd = True -> 2 entidades transferidas
    qtd = -2, normalize_qtd = False -> 5 entidades transferidas
  """
  if normalize_qtd: qtd = abs(qtd)

  # encontra os índices de ocorrências que possuem uma entidade em específico
  correcao_index = find_indexes_from_label(source, entity)[:qtd]

  for index in correcao_index:
    try:
      # utiliza iloc() para pegar a linha correspondentes no dataset origem
      row = source.iloc[index]
     
      # tenta remover a linha no dataset original
      source = source.drop(index)
      print(f"Dropped index: {index}")

      # tenta adicionar a linha no dataset de destino
      # esse comando só será executado caso a linha ainda não tenha sido transferida
      destination = destination.append(row, ignore_index=True)
      
    except KeyError:
      # falha ao remover uma linha, normalmente por que a linha já foi removida
      # (2 tags que devem ser balanceadas presentes na mesma linha)
      print(f"The index {index} was already dropped!")

  return destination, source




def split_percents(entidades_train, entidades_dev):
  """Encontra e retorna as proporções de distribuição
  das entidades nos datasets.

  Assume que o retorno da função 'count_entities' é uma
  lista de inteiros. Caso a função seja alterada para
  retornar um dicionário, esta função também deve ser
  alterada para manter compatibilidade.

  Parâmetros
  ----------
  entidades_train : list<int>
    Lista de inteiros contendo a contagem de ocorrências
    de cada entidade no dataset de treino, gerada através
    da função 'count_entities'.

  entidades_dev : list<int>
    Contagem de ocorrências das entidades no dataset dev.

  Retorno
  -------
  A função retorna uma tupla contendo três elementos:

  division_percent_train : list<float>
    Porcentagem de cada entidade no dataset de treino

  division_percent_dev : list<float>
    Porcentagem de cada entidade no dataset dev

  one_entity_percent_train : list<float>
    Relevância de uma única ocorrência de cada entidade em
    relação ao total de ocorrências daquela entidade nos
    datasets.
  """
  division_percent_train = []
  division_percent_dev = []
  one_entity_percent_train = []

  for train_percent, dev_percent in zip(entidades_train, entidades_dev):
    total = train_percent + dev_percent
    division_percent_train.append(train_percent / total if total > 0 else 0)
    division_percent_dev.append(1 - (train_percent / total) if total > 0 else 0)
    one_entity_percent_train.append(1 / total if total > 0 else 0)

  return division_percent_train, division_percent_dev, one_entity_percent_train


def get_balancing_samples(
  division_percent_train, division_percent_dev, one_division_percent_train,
  upper_limit=0.75, balancing_range=0.10):
  """Analisa as porcentagens geradas por 'split_percents' e calcula
  a quantidade de cada entidade que deve ser passada de um dataset
  para o outro.

  A combinação dos valores de 'upper_limit' e 'balancing_range' ditam
  como o balanceamento será feito. O valor do limite inferior é obtido
  através da operação 'upper_limit - balancing_range'.

  Parâmetros
  ----------
  upper_limit : float
    Um valor entre 0 e 1. Determina a porcentagem máxima de cada entidade
    no dataset de treino em comparação com o dataset dev.

  balancing_range : float
    Determina a proporção aceitável da distribuição de cada entidade nos
    datasets.

  Retorno
  -------
  Retorna uma lista de inteiros, onde cada elemento representa a quantidade
  de ocorrências daquela entidade que devem ser passadas de um dataset para
  o outro. Exemplo:
    +2: duas entidades devem ser passadas de train para dev
    -4: quatro entidades devem ser passadas de dev para train

  Nota
  ----
  O parâmetro 'division_percent_dev' foi mantido para garantir compatibilidade
  com outras funções e possibilidade de mudanças futuras.
  """
  balancing_samples = []
  lower_limit = upper_limit - balancing_range

  for train, dev, step in zip(division_percent_train, division_percent_dev, one_division_percent_train):
    c = 0 # quantas amostras do treino serão passadas para dev

    if train >= upper_limit:
      while train >= upper_limit:
        train -= step
        c += 1

    elif train <= lower_limit:
      while train <= lower_limit:
        train += step
        c -= 1

    balancing_samples.append(c)

  return balancing_samples


def count_entities(dataset, entities):
  """Conta o número de ocorrências de cada entidade
  em 'entities' no dataset.

  A função pode ser melhorada para retornar um dict
  com a relação < dict['entidade'] = ocorrências >.

  Porém, para evitar problemas de compatibilidade com
  outras funções, foi decidido manter o retorno como
  uma lista de inteiros.
  """
  entity_count = []
  for entity in entities:
    entity_count.append(dataset.loc[dataset[1]==entity, 1].count())

  return entity_count

def realizar_correcao(dataset_train, dataset_dev, contagem_correcao, nomes_entidades):
  """Balanceia um dataset com múltiplas classes (exemplo: dataset NER).

  A função não modifica os dataframes passados como argumento. Para garantir que
  eles sejam modificados, deve-se armazenar o valor de retorno da função.

  Exemplo:
    train, test = realizar_correcao(train, test, contagem, nomes)

  Parâmetros
  ----------
  dataset_train : pandas.DataFrame
    Subset de treino.

  dataset_train : pandas.DataFrame
    Subset de teste ou validação.

  contagem_correcao : list[int]
    Lista gerada pela função get_balancing_samples, contendo o número de entidades
      que devem ser passadas de um subset para o outro.
    Valores positivos indicam que as entidades devem ser passadas de train
      para dev; valores negativos indicam que devem ser passadas de dev para train.
    O número de elementos de 'contagem_correcao' deve ser igual ao número de elementos
      de 'nomes_entidades'.

  nomes_entidades : list
    Lista contendo os nomes das entidades. Deve ter o mesmo tamanho que
      'contagem_correcao'.

  Retorno
  -------
  Retorna uma tupla contendo os subsets modificados.
  """
  for correcao, entidade in zip(contagem_correcao, nomes_entidades):
    # passa amostras de treino pra dev
    if correcao > 0:
      dataset_dev, dataset_train = balance_entity(dataset_dev, dataset_train,
                                                    correcao, entidade)

    # passa amostras de dev pra treino
    elif correcao < 0:
      dataset_train, dataset_dev = balance_entity(dataset_train, dataset_dev,
                                                    correcao, entidade)

  return dataset_train, dataset_dev