import numpy as np
import scipy as sp
import sklearn

if __name__ == "__main__":
    classification_data = sklearn.datasets.make_classification()
    print(classification_data)

    # format danych to grupy punktow wygenerowanych z odchyleniem standardowym = 1 (rozklad normalny)
    # podzielonych na klasy
       
    training_indices = int(0.8 * len(classification_data)) - 1
    test_indices = len(classification_data) - training_indices
    
    data = classification_data.data
    
    training_data_part, test_data_part = sklearn.model_selection.train_test_split(data, test_size=0.2)
    print(training_data_part)
    print(test_data_part)
