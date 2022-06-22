import pickle
import multiprocessing

import pandas as pd

from dataset_generator import DatasetGenerator
from statistical_metrics_calculator import StatisticalMetricsCalculator
from metadataset_generator import MetaDatasetGenerator

# Features dict to choose parameters for generating dataframes.
features_dict = {
    "n_samples": [100, 1000, 5000, 10000],
    "n_features": [10, 50, 100],
    "n_informative": [2, 5, 8],
    "n_redundant": [2, 5, 8],
    "n_repeated": [0],
    "n_classes": [2, 8, 32],
    "n_clusters_per_class": [1, 2, 3, 5],
    # "weights": [[0.2], [0.6], [0.8]],
    "flip_y": [0.01, 0.05, 0.1, 0.2],
    "class_sep": [1, 2, 5, 10],
    "hypercube": [True, False]
}


# Generates 1 row of metadataframe,
# the function handles dataset generation, statistical metrics calculation and metadata creation.
# returns meta_df_dict: a dictionary that contains index as key and list of metadata X,metadata y,
# chosen metrics from data generation as value
# returns meta_matrix_dict: similar structure as meta_df_dict however,
# X and y are generated initial dataset. We store the initial data for backup.

def generate_meta_dataset(index):
    meta_df_dict = {}
    meta_matrix_dict = {}

    while True:

        dge = DatasetGenerator(features_dict=features_dict)
        X, y, random_feature_dict = dge.generate_dataset()

        if X is not None:
            break

    df, df_label, _ = dge.convert_dataset_to_dataframe()

    meta_matrix_dict[index] = [X, y, random_feature_dict]

    smc = StatisticalMetricsCalculator(generated_dataframe=df,
                                       generated_label_dataframe=df_label,
                                       generated_dataset=X)

    smc.prepare_statistical_metrics_dict()
    stats_metrics_dict = smc.get_statistical_metrics_dict()

    mdg = MetaDatasetGenerator(generated_dataframe=df,
                               generated_label_dataframe=df_label,
                               statistical_metrics=stats_metrics_dict)

    mdg.generate_metadata_dict()
    meta_data_dict = mdg.get_metadata_dict()
    meta_df_dict[index] = meta_data_dict

    df_meta_data = pd.DataFrame(meta_df_dict).T

    metadata_multi_label_matrix = mdg.get_metadata_multi_label_matrix()

    filehandler = open(f"./label_dict/5/multi_label_dict.pickle", "wb")
    pickle.dump(mdg.get_label_dict(), filehandler)
    filehandler.close()

    filehandler = open(f"./metadata_multi_label_matrices/5/metadata_multi_label_matrix_{index}.pickle", "wb")
    pickle.dump(metadata_multi_label_matrix, filehandler)
    filehandler.close()

    df_meta_data.to_csv(f'./metadataframes/5/meta_dataframe_{index}.csv')

    return None


if __name__ == '__main__':
    pool = multiprocessing.Pool()
    result_async = [pool.apply_async(generate_meta_dataset, args=(i,)) for i in
                    range(10000)]
    results = [r.get() for r in result_async]
