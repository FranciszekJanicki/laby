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
    P_cat = np.sum(P_cav_too_cat, axis=(0, 1))
    P_too_giv_cat = P_cav_too_cat / P_cat
    return P_too_giv_cat

def main():
    P_cav_too_cat = np.array([[[0.108, 0.012], [0.072, 0.008]],
                            [[0.016, 0.064], [0.144, 0.576]]])
        
    zadania = [zad1_2, zad1_3, zad1_4]    
    for zad_num, zad_result in enumerate(zadania):
        print("zadanie ", zad_num + 1, ": ", zad_result(P_cav_too_cat))

if __name__ == '__main__':
    main()
