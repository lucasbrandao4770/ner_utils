import os
from sklearn.model_selection import KFold
import utils 
import parse
from stats import DatasetAnalysis

args = parse.parseArguments() # PARSING ARGS

FILENAME = args.f
N_KFOLD = args.kfold
FOLDER_VERSION = args.version
MAX_LENGTH = args.max_length
assert os.path.exists(FOLDER_VERSION) == False, 'The version already exists' # assert the version do not exists
os.makedirs(FOLDER_VERSION)



print('Loading Dataset')
df = utils.conll2pandas(FILENAME) # LOAD THE DATASET FROM CONLL FILE

print('Dataset loaded')
kf = KFold(n_splits=N_KFOLD, random_state=0, shuffle=True) # KFOLD WITH SEED 0


# DATASET ANALYSIS
analysis = DatasetAnalysis(path = FILENAME, df=df)
stats = analysis.generate_dataset_info(is_alldata=True)
with open(os.path.join(FOLDER_VERSION, 'stats_full.txt'), 'w', encoding='utf-8') as f:
    f.writelines(stats)


analysis.convert_stats2excel(FOLDER_VERSION)
analysis.plot_graphs(FOLDER_VERSION)

# FILTER TAGS WITH MINIMUM RATIO
df = utils.filter_entities(df, minimum_entity_ratio=0.005) # removendo abaixo de 0.5%


# FILTER MAX_LENGHT SENTENCES
df = utils.filter_length_dataset(df, length_to_filter = MAX_LENGTH)  # FILTER SENTENCES TOO LONG


for i, (train_index, test_index) in enumerate(kf.split(df)): 
    save_path =  FOLDER_VERSION+'/'+'fold-'+str(i)+'/'    # PATH TO SAVE
    os.makedirs(save_path)    # CREATE THE FOLDER VERSION AND SUBFOLDER

        
    train_data, test_data = df.loc[train_index], df.loc[test_index] # get the data from indexes
    stats = []
    analysis = DatasetAnalysis(path = FILENAME, df=df)     
    stats.extend(analysis.generate_dataset_info(n_fold=i, train_data=True))  # TRAIN DATA
    stats.extend(analysis.generate_dataset_info(n_fold=i, train_data=False))  # TEST DATA
    
    # save stats    
    # WRITE FILE      
    with open(os.path.join(save_path, 'stats.txt'), 'w', encoding='utf-8') as f:
        f.writelines(stats)
    
    
    # CONVERT TO CONLL AND SAVE
    utils.pandas2conll(train_data, save_path+'train.conll')  # SAVE IN CONLL
    utils.pandas2conll(test_data, save_path+'dev.conll') 
    utils.pandas2json(train_data, save_path+'train.json')  # SAVE IN json
    utils.pandas2json(test_data, save_path+'dev.json') 
    print(f'Save dataset and stats for fold-{i}')

print('Done!')