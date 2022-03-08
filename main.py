import os
from sklearn.model_selection import KFold
import utils 
import generate_stats
import parse
from analysis_dataset import DatasetAnalysis

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
stats = analysis.generate_dataset_info(df, is_alldata=True)
f = open(os.path.join(FOLDER_VERSION, 'stats_full.txt'), 'w', encoding='utf-8')
f.writelines(stats)
f.close()
analysis.convert_stats2excel(FOLDER_VERSION)
analysis.plot_graphs(FOLDER_VERSION)

for i, (train_index, test_index) in enumerate(kf.split(df)): 
    save_path =  FOLDER_VERSION+'/'+'fold-'+str(i)+'/'    # PATH TO SAVE
    os.makedirs(save_path)    # CREATE THE FOLDER VERSION AND SUBFOLDER

        
    train_data, test_data = df.loc[train_index], df.loc[test_index] # get the data from indexes
    stats = []
    analysis = DatasetAnalysis(path = FILENAME, df=df)     
    stats.extend(analysis.generate_dataset_info(train_data, n_fold=i, train_data=True))  # TRAIN DATA
    stats.extend(analysis.generate_dataset_info(test_data, n_fold=i, train_data=False))  # TEST DATA
    
    # save stats    
    # WRITE FILE    
    f = open(os.path.join(save_path, 'stats.txt'), 'w', encoding='utf-8')
    f.writelines(stats)
    f.close()
    
    # FILTER MAX_LENGHT SENTENCES
    train_data = utils.filter_length_dataset(train_data, length_to_filter = MAX_LENGTH) # FILTER SENTENCES TOO LONG
    test_data  = utils.filter_length_dataset(test_data, length_to_filter = MAX_LENGTH) # FILTER SENTENCES TOO LONG
    
    # CONVERT TO CONLL AND SAVE
    utils.pandas2conll(train_data, save_path+'train.conll')  # SAVE IN CONLL
    utils.pandas2conll(test_data, save_path+'dev.conll') 
    print(f'Save dataset and stats for fold-{i}')

print('Done!')