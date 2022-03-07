# CROSS VALIDATION SPLIT AND STATISTICS
Splits NER dataset into cross validation and generate statistics.

Generates subfolder for each version and fold's, with stats and filtered max_length sentence.


## How to use 
- Example

python main.py --f corejur_nerv2.conll --kfold 5 --version versao0 --max_length 250



## Args 
Args:

    --f filename (dataset.conll)
    
    --kfold kfold's number (5)
    
    --version folder_version (version_0)
    
    --max_length filter for sentence tokens max_length (250)
    
