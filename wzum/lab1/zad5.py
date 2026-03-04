import numpy as np
import scipy as sp
import sklearn

if __name__ == "__main__":
    housing_data = sklearn.datasets.fetch_california_housing()
    print(housing_data)

    # format danych to obiekt gdzie poszczególne data-membery (składowe typu aggregate), 
    # to odpowiednio array data, array target oraz lista stringow feature_names
    
    data = housing_data.data
    target = housing_data.target
    feature_names = housing_data.feature_names
    
    training_indices = int(0.8 * len(data)) - 1
    test_indices = len(data) - training_indices

    training_data_part, test_data_part = sklearn.model_selection.train_test_split(data, test_size=0.2)
    print(training_data_part)
    print(test_data_part)
