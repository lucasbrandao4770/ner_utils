import balanceamento
import os
from sklearn.model_selection import KFold
from src import utils
from src import parse
from src.stats import DatasetAnalysis
import src.dataset_preprocessing as preprocessing

args = parse.parseArguments()  # PARSING ARGS

FILENAME = args.f
N_KFOLD = args.kfold
FOLDER_VERSION = "_"+args.version  # cause of gitignore
MAX_LENGTH = args.max_length
VERBOSE = args.verbose
# assert the version do not exists
assert os.path.exists(FOLDER_VERSION) == False, 'The version already exists'
os.makedirs(FOLDER_VERSION)


print('Loading Dataset')
df = utils.conll2pandas(FILENAME)  # LOAD THE DATASET FROM CONLL FILE
print('Dataset loaded')


####################### ALL DATA ANALYSIS #######################

analysis_fulldataset = DatasetAnalysis(path=FILENAME, df=df)
stats = analysis_fulldataset.generate_dataset_info(is_alldata=True)
with open(os.path.join(FOLDER_VERSION, 'stats_full.txt'), 'w', encoding='utf-8') as f:
    f.writelines(stats)

analysis_fulldataset.convert_stats2excel(FOLDER_VERSION)
if VERBOSE:
    analysis_fulldataset.plot_graphs(FOLDER_VERSION)


####################### PRE PROCESSING #######################

# FILTER TAGS WITH MINIMUM RATIO
# df = utils.filter_entities(df, minimum_entity_ratio=0.005) # removendo abaixo de 0.5%

# FILTER SENTENCES WITH ENTITIES
entities_to_remove = ['CNPJ', 'CPF', 'CNPJ_do_autor', 'CPF_do_réu']
df = preprocessing.remove_entites(df, entities_to_remove)

# FILTER MAX_LENGHT SENTENCES
df = preprocessing.filter_length_dataset(
    df, length_to_filter=MAX_LENGTH)  # FILTER SENTENCES TOO LONG

# UNDERSAMPLING SENTENCES WITH FULL 'O' TAGS
#df = preprocessing.undersampling_null_sentences(df, ratio_to_remove=0.8)

# UNDERSAMPLING TAGs
undersampling_tags = ['Normativo']
df = preprocessing.undersampling_entity(
    df, undersampling_tags=undersampling_tags, ratio_to_remove=0.8)


####################### SPLIT IN K FOLDS AND GENERATE ANALYSIS #######################
kf = KFold(n_splits=N_KFOLD, random_state=0, shuffle=True)  # KFOLD WITH SEED 0

for i, (train_index, test_index) in enumerate(kf.split(df)):
    save_path = FOLDER_VERSION+'/'+'fold-'+str(i)+'/'    # PATH TO SAVE
    os.makedirs(save_path)    # CREATE THE FOLDER VERSION AND SUBFOLDER

    # get the data from indexes
    train_data, test_data = df.loc[train_index], df.loc[test_index]

    # FOLD ANALYSIS
    stats = []
    analysis_train = DatasetAnalysis(path=FILENAME, df=train_data)
    analysis_test = DatasetAnalysis(path=FILENAME, df=test_data)
    stats.extend(analysis_train.generate_dataset_info(
        n_fold=i, train_data=True))  # TRAIN DATA
    stats.extend(analysis_test.generate_dataset_info(
        n_fold=i, train_data=False))  # TEST DATA

    # save stats
    # WRITE FILE
    with open(os.path.join(save_path, 'stats.txt'), 'w', encoding='utf-8') as f:
        f.writelines(stats)

    # CONVERT TO CONLL AND SAVE
    utils.pandas2conll(train_data, save_path+'train.conll')  # SAVE IN CONLL
    utils.pandas2conll(test_data, save_path+'dev.conll')

    # BALANCE AND REWRITE CONLL FILES
    train_data, test_data = balanceamento.balance_from_conll(save_path+'train.conll',
                                                             save_path+'dev.conll')

    # SAVE AGAIN IN CONLL AND JSON
    utils.pandas2conll(train_data, save_path+'train.conll')  # SAVE IN CONLL
    utils.pandas2conll(test_data, save_path+'dev.conll')
    utils.pandas2json(train_data, save_path+'train.json')  # SAVE IN JSON
    utils.pandas2json(test_data, save_path+'dev.json')

    print(f'Save dataset and stats for fold-{i}')

print('Done!')
