import numpy as np
import scipy as sp
import sklearn
from sklearn.datasets import fetch_openml

if __name__ == "__main__":
    openml_data = sklearn.datasets.fetch_openml(name='miceprotein')
    print(openml_data)

    # format danych to mapa (array par kluczy-wartosci), gdzie klucze to data, wartosc dla niego to array danych,
    # target dla którego wartość to etykiety, klucz descr ma wartosc stringa z opisem danych, natomiast klucze feature_names i 
    # list_names mapuja na wartosci list stringow przechowujących nazwy kolumn danych oraz nazwy kolumn etykiet
    
    # zawiera dane dotyczące poziomów białek w myszach. dane pochodzą z eksperymentów biologicznych
    # celem jest klasyfikacja oparta na wartości pomiarów protein
       
    training_indices = int(0.8 * len(openml_data)) - 1
    test_indices = len(openml_data) - training_indices
    
    data = openml_data.data
    
    training_data_part, test_data_part = sklearn.model_selection.train_test_split(data, test_size=0.2)
    print(training_data_part)
    print(test_data_part)
