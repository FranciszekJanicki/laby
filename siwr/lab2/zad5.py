#!/usr/bin/env python

import numpy as np


def normalize(v):
    return v / np.sum(v)


def forward(prev, evidence, transition, sensor):
    pred = np.dot(transition.T, prev)
    updated = sensor[evidence] * pred
    return normalize(updated)


def main():
    # stan: [Rain=true, Rain=false]

    # P(R_t+1 | R_t)
    transition = np.array([
        [0.7, 0.3],
        [0.3, 0.7]
    ])

    # P(Umbrella | Rain)
    sensor = {
        True: np.array([0.9, 0.2]),
        False: np.array([0.1, 0.8])
    }

    # poczatkowy rozkLad
    belief = np.array([0.5, 0.5])

    observations = [True, True, False, True, True]

    results = []

    for obs in observations:
        belief = forward(belief, obs, transition, sensor)
        results.append(belief[0])  # P(Rain=true)

    print("Kolejne dni:")
    print(results)

    print("\nP(Rain_5 | obserwacje) =", results[-1])

    # predykcja (bez obserwacji)
    belief6 = np.dot(transition.T, belief)
    belief9 = belief6
    for _ in range(3):
        belief9 = np.dot(transition.T, belief9)

    print("\nPredykcja:")
    print("t=6:", belief6[0])
    print("t=9:", belief9[0])


if __name__ == "__main__":
    main()