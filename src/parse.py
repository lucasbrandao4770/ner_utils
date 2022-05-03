import argparse

def parseArguments():
    parser = argparse.ArgumentParser(description='Split NER dataset into KFolds and generate stats on it')
    parser.add_argument("--f", type=str, help="Dataset filename -> example: data.conll")
    parser.add_argument("--kfold", type=int, default=5, help="Number of K-folds")
    parser.add_argument("--version", type=str,default=None, help="Version of dataset, and filename for saving")
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,       
        help="Max tokens length for sentence")
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose for showing the graphs')

    args, _ = parser.parse_known_args()
        
    
    return args