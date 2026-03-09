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

def zad1_6(n: int):
    return 2 * n

def zad1_7(n):
    return zad1_6(n) * sizeof(float)  
 
def zad1_8(P_cav_too_cat : np.array):
    P_cav_too = np.sum(P_cav_too_cat, axis=-1)
    P_cav = np.sum(P_cav_too, axis=0)
    P_too = np.sum(P_cav_too, axis=1)
    P_cat = np.sum(P_cav_too, axis=(0, 1))
    P_too_cat_giv_cav = P_too_giv_cat * P_too_giv_cav
    P_cav_giv_too = P_cav_too / P_too
    P_cav_giv_too_cat = None
    # P_too_cat = P(a), P_cat = P(b) => P_too_cat = P(a, b) = P(a | b) * P(b) = P(too | cat) * P(cat)
    P_cav_giv_too_cat_nn = np.transpose(P_too_cat_giv_cav * P_cav, (2, 0, 1))
    alpha_too_cat = 1 / np.sum(P_cav_giv_too_cat_nn, axis=0)
    P_cav_giv_too_cat = P_cav_giv_too_cat_nn * alpha_too_cat
    return P_cav_giv_too_cat

def main():
    P_cav_too_cat = np.array([[[0.108, 0.012], [0.072, 0.008]],
                            [[0.016, 0.064], [0.144, 0.576]]])
        
    zadania = [zad1_2, zad1_3, zad1_4, zad1_5, zad1_6, zad1_7, zad1_8]    
    for zad_num, zad_result in enumerate(zadania):
        print("zadanie ", zad_num + 1, ": ", zad_result(P_cav_too_cat))

if __name__ == '__main__':
    main()
