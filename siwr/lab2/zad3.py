#!/usr/bin/env python

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from itertools import product
import numpy as np


def main():
    # MODEL
    model = BayesianModel([
        ('Cavity', 'Toothache'),
        ('Cavity', 'Catch')
    ])

    # CPD z nazwami stanow
    cpd_cavity = TabularCPD(
        variable='Cavity',
        variable_card=2,
        values=[[0.2], [0.8]],
        state_names={'Cavity': ['true', 'false']}
    )

    cpd_toothache = TabularCPD(
        variable='Toothache',
        variable_card=2,
        values=[[0.6, 0.1],
                [0.4, 0.9]],
        evidence=['Cavity'],
        evidence_card=[2],
        state_names={
            'Toothache': ['true', 'false'],
            'Cavity': ['true', 'false']
        }
    )

    cpd_catch = TabularCPD(
        variable='Catch',
        variable_card=2,
        values=[[0.9, 0.2],
                [0.1, 0.8]],
        evidence=['Cavity'],
        evidence_card=[2],
        state_names={
            'Catch': ['true', 'false'],
            'Cavity': ['true', 'false']
        }
    )

    model.add_cpds(cpd_cavity, cpd_toothache, cpd_catch)

    print("Model OK:", model.check_model())

    infer = VariableElimination(model)

    # P(Cavity | Toothache=true, Catch=false)
    q = infer.query(['Cavity'],
                    evidence={'Toothache': 'true', 'Catch': 'false'})
    print("\nP(Cavity | Toothache=true, Catch=false):")
    print(q)

    # pelna tabela
    print("\nPelna tabela P(Cavity | Toothache, Catch):")
    for t, c in product(['true', 'false'], repeat=2):
        q = infer.query(['Cavity'],
                        evidence={'Toothache': t, 'Catch': c})
        print(f"Toothache={t}, Catch={c} -> {q.values}")


if __name__ == '__main__':
    main()