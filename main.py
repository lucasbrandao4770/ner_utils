import os
from sklearn.model_selection import KFold
import utils 
import generate_stats
import parse


args = parse.parseArguments() # PARSING ARGS

FILENAME = args.f
N_KFOLD = args.kfold
FOLDER_VERSION = args.version
MAX_LENGHT = args.max_lenght

assert os.path.exists(FOLDER_VERSION) == False, 'The version already exists' # assert the version do not exists


print('Loading Dataset')
df = utils.conll2pandas(FILENAME) # LOAD THE DATASET FROM CONLL FILE

print('Dataset loaded')
kf = KFold(n_splits=N_KFOLD, random_state=0, shuffle=True)


stats = generate_stats.generate_dataset_info(df, is_alldata=True) # GENERATE STATS FOR ALL DATA
# WRITE FILE    
f = open(os.path.join('./', 'stats.txt'), 'w', encoding='utf-8')
f.writelines(stats)
f.close()
print('Saved stats on all dataset')

for i, (train_index, test_index) in enumerate(kf.split(df)): 
    save_path =  FOLDER_VERSION+'/'+'fold-'+str(i)+'/'    # PATH TO SAVE
    os.makedirs(save_path)    # CREATE THE FOLDER VERSION AND SUBFOLDER
    
    train_data, test_data = df.loc[train_index], df.loc[test_index] # get the data from indexes
    stats = []
    stats.extend(generate_stats.generate_dataset_info(train_data, n_fold=i, train_data=True)) # get the stats from train data
    stats.extend(generate_stats.generate_dataset_info(test_data, n_fold=i, train_data=False)) # get the stats from test data
    
    # save stats    
    # WRITE FILE    
    f = open(os.path.join(save_path, 'stats.txt'), 'w', encoding='utf-8')
    f.writelines(stats)
    f.close()
    
    # FILTER MAX_LENGHT SENTENCES
    train_data = train_data[train_data['quantidadeTokens'] <= MAX_LENGHT]
    test_data  =  test_data[test_data['quantidadeTokens'] <= MAX_LENGHT]
    
    # CONVERT TO CONLL AND SAVE
    utils.pandas2conll(train_data, save_path+'train.conll')  # SAVE IN CONLL
    utils.pandas2conll(test_data, save_path+'dev.conll') 
    print(f'Saved dataset and stats for fold-{i}')
print('Done!')