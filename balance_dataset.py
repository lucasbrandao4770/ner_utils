from src import utils
from src.balanceamento import balance_from_conll, balance_from_one_conll
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    DATA_PATH = "corejur_ner_v20.conll"
    SAVE_PATH = "test/"

    # df = utils.conll2pandas(DATA_PATH)
    # train_df, test_df = train_test_split(df, test_size=0.2)
    # utils.pandas2conll(train_df, SAVE_PATH+'train.conll')  # SAVE IN CONLL
    # utils.pandas2conll(test_df, SAVE_PATH+'dev.conll')

    # TRAIN_PATH = SAVE_PATH+"train.conll"
    # TEST_PATH = SAVE_PATH+"dev.conll"

    train_df_balanced, test_df_balanced = balance_from_one_conll(DATA_PATH, test_size=0.2)

    utils.pandas2json(train_df_balanced, SAVE_PATH+'train_balanceado.json')  # SAVE IN JSON
    utils.pandas2json(test_df_balanced, SAVE_PATH+'dev_balanceado.json')
