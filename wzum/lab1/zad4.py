import numpy as np
import scipy as sp
import sklearn

if __name__ == "__main__":
    faces_data = sklearn.datasets.fetch_olivetti_faces()
    print(faces_data)

    # format danych to mapa (array key-value par), gdzie klucze to stringi, natomiast 
    # wartisci to macierze (tablice tablic)
    
    data = faces_data["data"]
    target = faces_data["target"]
    images = faces_data["images"]
    
    training_indices = int(0.8 * len(data)) - 1
    test_indices = len(data) - training_indices
    
    training_data_part, test_data_part = sklearn.model_selection.train_test_split(data, test_size=0.2)
    print(training_data_part)
    print(test_data_part)
