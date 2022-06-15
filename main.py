import os
from sklearn.model_selection import KFold
from src import utils
from src.stats import DatasetAnalysis
import src.dataset_preprocessing as preprocessing
from src.balanceamento import balance_from_conll
from configparser import ConfigParser
from src.utils import fix_seed


config = ConfigParser()
config.read('settings.cfg')
random_state = config['UTILS'].getint('random_state', 0)
fix_seed(random_state)


FILENAME = os.path.join(config['DATASET']['folder'],
                        config['DATASET']['filename'])
N_KFOLD = config['KFOLD'].getint('n_fold', 5)  # DEFAULT VALUE OF 5
# _ because of gitignore
SAVE_FOLDER = '_'+config['SAVE'].get('save_folder', 'output_folder')

# assert the version do not exists
assert os.path.exists(SAVE_FOLDER) == False, 'The version already exists'
os.makedirs(SAVE_FOLDER)


print('Loading Dataset')
df = utils.conll2pandas(FILENAME)  # LOAD THE DATASET FROM CONLL FILE
print('Dataset loaded')


####################### ALL DATA ANALYSIS #######################

analysis_fulldataset = DatasetAnalysis(df=df)
stats = analysis_fulldataset.generate_dataset_info(is_alldata=True)
with open(os.path.join(SAVE_FOLDER, 'stats_full.txt'), 'w', encoding='utf-8') as f:
    f.writelines(stats)

analysis_fulldataset.convert_stats2excel(SAVE_FOLDER)
analysis_fulldataset.plot_graphs(
    SAVE_FOLDER, verbose=config['UTILS'].getboolean('plot_verbose', False))


####################### PRE PROCESSING #######################
print('Preprocessing dataset')

# FILTER TAGS WITH MINIMUM RATIO
# df = utils.filter_entities(df, minimum_entity_ratio=0.005) # removendo abaixo de 0.5%

# FILTER SENTENCES WITH ENTITIES
tags_to_remove = config['PREPROCESSING'].get('fill_O_tags', '')
if tags_to_remove:
    # e.g ['CNPJ', 'CPF', 'CNPJ_do_autor', 'CPF_do_réu']
    tags_to_remove = tags_to_remove.split(', ')
df = preprocessing.fill_O_tags(df, tags_to_remove)

# Datas_do_contrato e Datas_dos_fatos PARA Datas
if config['PREPROCESSING'].getboolean('datas_aggregation', True):
    print('Datas Aggretation')
    # hardcoded due to business decision
    datas_to_change = ['Data_do_contrato', 'Data_dos_fatos']
    df = preprocessing.datas_change(df, datas_to_change=datas_to_change)


if config['PREPROCESSING'].getboolean('remove_jurisprudencia_sentence', True):
    print('Remove Jurisprudência')
    # REMOVE JURISPRUDENCIA
    df = preprocessing.remove_jurisprudencia_sentence(df)


# A MUST STEP
# FILTER MAX_LENGHT SENTENCES
df = preprocessing.trucate_sentence_max_length(
    df, max_length=config['PREPROCESSING'].getint('max_length_sentence', 256))


# UNDERSAMPLING TAGs

undersampling_tags = config['PREPROCESSING'].get('undersampling_tags', '')
if undersampling_tags:
    undersampling_tags = undersampling_tags.split(' ')  # ['Normativo']
    print('Undersampling tags ', undersampling_tags)
    df = preprocessing.undersampling_entity(
        df, undersampling_tags=undersampling_tags,
        ratio_to_remove=config['PREPROCESSING'].getfloat(
            'ratio_of_undersample_tags', 0.5)
    )


print('SPLITS into FOLDS')

####################### SPLIT IN K FOLDS AND GENERATE ANALYSIS #######################
kf = KFold(n_splits=N_KFOLD, random_state=random_state, shuffle=True)  # KFOLD

for i, (train_index, test_index) in enumerate(kf.split(df)):
    save_path = SAVE_FOLDER+'/'+'fold-'+str(i)+'/'    # PATH TO SAVE
    os.makedirs(save_path)    # CREATE THE FOLDER VERSION AND SUBFOLDER

    # get the data from indexes
    train_data, test_data = df.loc[train_index], df.loc[test_index]

    # UNDERSAMPLING SENTENCES WITH FULL 'O' TAGS
    # ONLY IN TRAIN
    if config['PREPROCESSING'].get('undersampling_negative_sentences'):
        train_data = preprocessing.undersampling_negative_sentences(
            train_data,
            ratio_to_remove=config['PREPROCESSING'].getfloat(
                'ratio_of_undersample_negative_sentences', 0.8)
        )

    # FOLD ANALYSIS
    stats = []
    analysis_train = DatasetAnalysis(df=train_data)
    analysis_test = DatasetAnalysis(df=test_data)
    stats.extend(analysis_train.generate_dataset_info(
        n_fold=i, train_data=True))  # TRAIN DATA
    stats.extend(analysis_test.generate_dataset_info(
        n_fold=i, train_data=False))  # TEST DATA
    # save stats
    # WRITE FILE
    with open(os.path.join(save_path, 'stats.txt'), 'w', encoding='utf-8') as f:
        f.writelines(stats)

    # SAVE KFOLD SPLIT DATASET
    if config['SAVE'].get('save_into_conll', True):
        utils.pandas2conll(train_data, save_path +
                           'train.conll')  # SAVE IN CONLL
        utils.pandas2conll(test_data, save_path+'dev.conll')
    if config['SAVE'].get('save_into_json', True):
        utils.pandas2json(train_data, save_path+'train.json')  # SAVE IN JSON
        utils.pandas2json(test_data, save_path+'dev.json')

    if config['PREPROCESSING'].get('balance_folds', True):
        # BALANCE AND REWRITE CONLL FILES
        train_data, test_data = balance_from_conll(save_path+'train.conll',
                                                   save_path+'dev.conll')

        # SAVE BALANCED DATASET
        utils.pandas2conll(train_data, save_path +
                           'train_balanceado.conll')  # SAVE IN CONLL
        utils.pandas2conll(test_data, save_path+'dev_balanceado.conll')
        utils.pandas2json(train_data, save_path +
                          'train_balanceado.json')  # SAVE IN JSON
        utils.pandas2json(test_data, save_path+'dev_balanceado.json')

    print(f'Save dataset and stats for fold-{i}')

    if config['SAVE'].getboolean('save_only_first_fold', True):
        print('SAVING ONLY FOLD 0')
        break

print('Done!')
