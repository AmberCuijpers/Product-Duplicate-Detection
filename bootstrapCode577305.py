import numpy as np
import pandas as pd


def get_training_test_data(dataframe, number_of_products):
    indices = dataframe.index.tolist()
    indices_training = np.random.choice(indices, size=number_of_products, replace=True)
    # Sort by index and if an index appears multiple times, drop the duplicates until it only appears once
    training_data = pd.DataFrame([dataframe.iloc[i] for i in indices_training]).sort_index()
    training_data_hashable = training_data.map(lambda x: hash(tuple(x)) if isinstance(x, set) else x)
    training_data = training_data.loc[~training_data_hashable.duplicated(keep='first')]
    training_data.reset_index(drop=True, inplace=True)

    indices_test = list(set(indices) - set(indices_training))
    test_data = pd.DataFrame([dataframe.iloc[i] for i in indices_test]).sort_index()
    test_data.reset_index(drop=True, inplace=True)
    return training_data, test_data
