"""
    Main file for Named Entity Recognition Utils
    Contains Dataset Stats, KFOLD splits, Fold Stratifed Balance
    All settings must be changed in config/settings.yaml folder

"""
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold

import src.dataset_preprocessing as preprocessing
from src import utils
from src.balanceamento import balance_from_conll
from src.stats import DatasetAnalysis
from src.utils import fix_seed


@hydra.main(config_path="config", config_name="settings")
def main(config: DictConfig):
    """Run the entire pipe
    Contains Stats, Kfold splits and fold balancing


    Args:
        config (DictConfig): All settings in settings.yaml file
    """

    random_state = config["UTILS"].get("random_state", 0)
    fix_seed(random_state)

    FILENAME = os.path.join(config["DATASET"]["folder"], config["DATASET"]["filename"])

    N_KFOLD = config["KFOLD"].get("n_fold", 5)  # DEFAULT VALUE OF 5
    # _ because of gitignore
    SAVE_FOLDER = config["SAVE"].get("save_folder", "output_folder")

    # assert the version do not exists
    assert os.path.exists(SAVE_FOLDER) is False, "The version already exists"
    os.makedirs(SAVE_FOLDER)

    print("Loading Dataset")
    df = utils.conll2pandas(FILENAME)  # LOAD THE DATASET FROM CONLL FILE
    print("Dataset loaded")

    # ---------------------- ALL DATA ANALYSIS ----------------------

    analysis_fulldataset = DatasetAnalysis(df=df)
    stats = analysis_fulldataset.generate_dataset_info(is_alldata=True)
    with open(os.path.join(SAVE_FOLDER, "stats_full.txt"), "w", encoding="utf-8") as f:
        f.writelines(stats)

    analysis_fulldataset.convert_stats2excel(SAVE_FOLDER)
    analysis_fulldataset.plot_graphs(
        SAVE_FOLDER, verbose=config["UTILS"].get("plot_verbose", False)
    )

    # ---------------------- PRE PROCESSING  ----------------------
    print("Preprocessing dataset")

    # FILTER TAGS WITH MINIMUM RATIO # removendo abaixo de 0.5%
    # df = utils.filter_entities(df, minimum_entity_ratio=0.005)

    # FILTER SENTENCES WITH ENTITIES
    tags_to_remove = config["PREPROCESSING"].get("fill_O_tags", "")
    if tags_to_remove:
        df = preprocessing.fill_O_tags(df, tags_to_remove)

    # Datas_do_contrato e Datas_dos_fatos PARA Datas
    datas_to_change = config["PREPROCESSING"].get("datas_aggregation")
    if datas_to_change:
        print("Datas Aggretation to Generic Datas", datas_to_change)
        # hardcoded due to business decision
        df = preprocessing.datas_change(df, datas_to_change=datas_to_change)

    if config["PREPROCESSING"].get("remove_jurisprudencia_sentence", False):
        print("Remove Jurisprudência")
        # REMOVE JURISPRUDENCIA
        df = preprocessing.remove_jurisprudencia_sentence(df)

    # A MUST STEP
    # FILTER MAX_LENGHT SENTENCES
    df = preprocessing.trucate_sentence_max_length(
        df, max_length=config["PREPROCESSING"].get("max_length_sentence", 256)
    )

    # UNDERSAMPLING SENTENCES WITH FULL 'O' TAGS
    # ONLY IN TRAIN
    if config["PREPROCESSING"].get("undersampling_negative_sentences"):
        print("UNDERSAMPLING NEGATIVE SENTENCES")
        df = preprocessing.undersampling_negative_sentences(
            df,
            ratio_to_remove=config["PREPROCESSING"].get(
                "ratio_of_undersample_negative_sentences", 0.8
            ),
        )

    # UNDERSAMPLING TAGs
    undersampling_tags = config["PREPROCESSING"].get("undersampling_tags")
    if undersampling_tags:
        print("Undersampling tags ", undersampling_tags)
        df = preprocessing.undersampling_entity(
            df,
            undersampling_tags=undersampling_tags,
            ratio_to_remove=config["PREPROCESSING"].get(
                "ratio_of_undersample_tags", 0.5
            ),
        )

    print("SPLITS into FOLDS")

    # --------------- SPLIT IN K FOLDS AND GENERATE ANALYSIS ----------
    # KFOLD
    kf = KFold(n_splits=N_KFOLD, random_state=random_state, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(df)):
        save_path = SAVE_FOLDER + "/" + "fold-" + str(i) + "/"  # PATH TO SAVE
        os.makedirs(save_path)  # CREATE THE FOLDER VERSION AND SUBFOLDER

        # get the data from indexes
        train_data, test_data = df.loc[train_index], df.loc[test_index]

        # FOLD ANALYSIS
        stats = []
        analysis_train = DatasetAnalysis(df=train_data)
        analysis_test = DatasetAnalysis(df=test_data)
        stats.extend(
            analysis_train.generate_dataset_info(n_fold=i, train_data=True)
        )  # TRAIN DATA
        stats.extend(
            analysis_test.generate_dataset_info(n_fold=i, train_data=False)
        )  # TEST DATA
        # save stats

        # SAVE KFOLD SPLIT DATASET
        # SAVE IN CONLL
        if config["SAVE"].get("save_into_conll", True):
            utils.pandas2conll(train_data, save_path + "train.conll")
            utils.pandas2conll(test_data, save_path + "dev.conll")
        # SAVE IN JSON
        if config["SAVE"].get("save_into_json", True):
            utils.pandas2json(train_data, save_path + "train.json")
            utils.pandas2json(test_data, save_path + "dev.json")

        if config["PREPROCESSING"].get("balance_folds", True):
            print("BALANCING FOLD")
            # BALANCE AND REWRITE CONLL FILES
            train_data, test_data = balance_from_conll(
                save_path + "train.conll", save_path + "dev.conll"
            )

            # SAVE BALANCED DATASET
            # SAVE IN CONLL
            utils.pandas2conll(train_data, save_path + "train.conll")
            utils.pandas2conll(test_data, save_path + "dev.conll")
            # SAVE IN JSON
            utils.pandas2json(train_data, save_path + "train.json")
            utils.pandas2json(test_data, save_path + "dev.json")

            stats.append("*" * 15)
            stats.append("STATS WITH FOLDS BALANCED")
            stats.append("*" * 15 + "\n")

            analysis_train = DatasetAnalysis(df=train_data)
            analysis_test = DatasetAnalysis(df=test_data)
            stats.extend(
                analysis_train.generate_dataset_info(n_fold=i, train_data=True)
            )  # TRAIN DATA
            stats.extend(
                analysis_test.generate_dataset_info(n_fold=i, train_data=False)
            )  # TEST DATA

        with open(os.path.join(save_path, "stats.txt"), "w", encoding="utf-8") as f:
            f.writelines(stats)

        print(f"Save dataset and stats for fold-{i}")

        with open(
            os.path.join(save_path, "preprocessing_snapshot.yaml"),
            "w",
            encoding="utf-8",
        ) as f:
            f.writelines(OmegaConf.to_yaml(config["PREPROCESSING"]))

        if config["SAVE"].get("save_only_first_fold", True):
            print("SAVING ONLY FOLD 0")
            break

    print("Done!")


if __name__ == "__main__":
    main()
