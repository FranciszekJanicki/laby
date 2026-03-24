import numpy as np

def zad1_2(P_cav_too_cat : np.array):
    P_cav_too = np.sum(P_cav_too_cat, axis=-1)
    return P_cav_too

def zad1_3(P_cav_too_cat : np.array):
    P_cav_too = np.sum(P_cav_too_cat, axis=-1)
    P_cav = np.sum(P_cav_too, axis=0)
    return P_cav

def zad1_4(P_cav_too_cat : np.array):
    P_cav_too = np.sum(P_cav_too_cat, axis=-1)
    P_cav = np.sum(P_cav_too, axis=0)
    P_too = np.sum(P_cav_too, axis=1)
    P_cat = np.sum(P_cav_too, axis=(0, 1))
 
    # regula iloczynu
    # P(too, cav) = P(too | cav) * P(cav)
    # P(too | cav) = P(too, cav) / p(cav)
    P_too_giv_cav = P_cav_too_cat / P_cav
    return P_too_giv_cav

def zad1_5(P_cav_too_cat : np.array):
    P_cav_too = np.sum(P_cav_too_cat, axis=-1)
    P_cav = np.sum(P_cav_too, axis=0)
    P_too = np.sum(P_cav_too, axis=1)
    P_cat = np.sum(P_cav_too, axis=(0, 1))
    # regula bayesa
    # P(too, cav) = P(too | cav) * P(cav)
    # P(too v cat) = ?
    # P(cav | too v cat) = P(too, cav) / p(cav)
    # P(cav) = P(a), P(too v cat) = P(b) => cos tam cos tam
    return None

def zad1_6(P_cav_too_cat : np.array):
    return 2 * np.sum(P_cav_too_cat.shape)

def zad1_7(P_cav_too_cat : np.array):
    return zad1_6(P_cav_too_cat) * np.dtype(float).itemsize 
 
def zad1_8(P_cav_too_cat : np.array):
    P_cav_too = np.sum(P_cav_too_cat, axis=-1)
    P_cav = np.sum(P_cav_too, axis=0)
    P_too = np.sum(P_cav_too, axis=1)
    P_cat = np.sum(P_cav_too, axis=(0, 1))
    # P_too_cat_giv_cav = P_too_giv_cat * P_too_giv_cav
    # P_cav_giv_too = P_cav_too / P_too
    # P_cav_giv_too_cat = None
    # P_too_cat = P(a), P_cat = P(b) => P_too_cat = P(a, b) = P(a | b) * P(b) = P(too | cat) * P(cat)
    P_too_cat_giv_cav = np.transpose(P_cav_too_cat, (1, 2, 0)) / P_cav
    P_cav_giv_too_cat_nn = np.transpose(P_too_cat_giv_cav * P_cav, (2, 0, 1))
    alpha_too_cat = 1 / np.sum(P_cav_giv_too_cat_nn, axis=0)
    P_cav_giv_too_cat = P_cav_giv_too_cat_nn * alpha_too_cat
    print("P(cavity | toothache, catch):\n", P_cav_giv_too_cat, "\n")

def zad1_9(P_cav_too_cat : np.array):
    return "YES"
    
def zad1_10(P_cav_too_cat : np.array):
    P_cav_too = np.sum(P_cav_too_cat, axis=-1)
    P_cav = np.sum(P_cav_too, axis=0)
    P_too = np.sum(P_cav_too, axis=1)
    P_cat = np.sum(P_cav_too, axis=(0, 1))
    # Simulate that we have P_too_giv_cav and P_cat_giv_cav.
    P_too_giv_cav = np.transpose(P_cav_too) / P_cav
    P_cav_cat = np.sum(P_cav_too_cat, axis=1)
    P_cat_giv_cav = np.transpose(P_cav_cat) / P_cav
    # Use expand_dims to insert dummy dimensions, so each variable is broadcasted properly.
    P_too_cat_giv_cav = np.transpose(P_cav_too_cat, (1, 2, 0)) / P_cav
    P_cav_giv_too_cat_nn = np.transpose(P_too_cat_giv_cav * P_cav, (2, 0, 1))
    alpha_too_cat = 1 / np.sum(P_cav_giv_too_cat_nn, axis=0)
    P_cav_giv_too_cat_nn2 = np.transpose(np.expand_dims(P_too_giv_cav, axis=1) * np.expand_dims(P_cat_giv_cav, axis=0) * P_cav, (2, 0, 1))
    P_cav_giv_too_cat = P_cav_giv_too_cat_nn2 * alpha_too_cat
    return P_cav_giv_too_cat
    
def zad1_11(P_cav_too_cat : np.array):
    return None
    
def zad1_12(P_cav_too_cat : np.array):
    return None

def main():
    P_cav_too_cat = np.array([[[0.108, 0.012], [0.072, 0.008]],
                            [[0.016, 0.064], [0.144, 0.576]]])
        
    zadanias = [zad1_2, zad1_3, zad1_4, zad1_5, zad1_6, zad1_7, zad1_8, zad1_9, zad1_10, zad1_11, zad1_12]    
    for zad_num, zad_result in enumerate(zadanias):
        print("zadanie ", zad_num + 1, ": ", zad_result(P_cav_too_cat))

if __name__ == '__main__':
    main()
