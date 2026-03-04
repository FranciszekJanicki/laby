#!/usr/bin/env python3

import numpy as np

# P_cav = P(A)
# P_tooth = P(B)

def main():
    P1 =	[[0.108, 0.012], [0.016, 0.064]]
    P2 = [[0.072, 0.008], [0.144, 0.576]]
    
    P_cav = np.sum(P1)
    P_tooth = np.sum(P2)

    print((P_cav, P_tooth))

if __name__ == '__main__':
    main()
