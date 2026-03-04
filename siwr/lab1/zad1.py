#!/usr/bin/env python3

import numpy as np

def zad1_2(P_cav_too_cat : np.array):
    P_cav_too = np.sum(P_cav_too_cat, axis=-1)


def main():
    P_cav_too_cat = np.array([[[0.108, 0.012], [0.072, 0.008]],
                            [[0.016, 0.064], [0.144, 0.576]]])
        
    print((P_cav, P_too))

if __name__ == '__main__':
    main()
