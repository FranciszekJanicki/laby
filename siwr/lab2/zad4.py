#!/usr/bin/env python

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def main():
    model = BayesianModel([
        ('Battery', 'Starts'),
        ('Fuel', 'Starts'),
        ('StarterMotor', 'Starts'),
        ('NotIcyWeather', 'Starts'),
        ('Starts', 'Moves'),
        ('Battery', 'Radio')
    ])

    # CPD
    cpd_battery = TabularCPD('Battery', 2, [[0.9], [0.1]])
    cpd_fuel = TabularCPD('Fuel', 2, [[0.9], [0.1]])
    cpd_weather = TabularCPD('NotIcyWeather', 2, [[0.9], [0.1]])
    cpd_starter = TabularCPD('StarterMotor', 2, [[0.95], [0.05]])

    # Starts zalezy od 4 zmiennych (16 kombinacji!)
    # uproszczony model
    cpd_starts = TabularCPD(
        'Starts', 2,
        values=[
            [0.99, 0.8, 0.8, 0.5, 0.8, 0.5, 0.5, 0.2,
             0.8, 0.5, 0.5, 0.2, 0.5, 0.2, 0.2, 0.0],
            [0.01, 0.2, 0.2, 0.5, 0.2, 0.5, 0.5, 0.8,
             0.2, 0.5, 0.5, 0.8, 0.5, 0.8, 0.8, 1.0]
        ],
        evidence=['Battery', 'Fuel', 'StarterMotor', 'NotIcyWeather'],
        evidence_card=[2, 2, 2, 2]
    )

    cpd_moves = TabularCPD(
        'Moves', 2,
        values=[[0.99, 0.0],
                [0.01, 1.0]],
        evidence=['Starts'],
        evidence_card=[2]
    )

    cpd_radio = TabularCPD(
        'Radio', 2,
        values=[[0.9, 0.1],
                [0.1, 0.9]],
        evidence=['Battery'],
        evidence_card=[2]
    )

    model.add_cpds(cpd_battery, cpd_fuel, cpd_weather,
                   cpd_starter, cpd_starts, cpd_moves, cpd_radio)

    print("Model OK:", model.check_model())

    infer = VariableElimination(model)

    # P(Starts | Radio=true, Fuel=true)
    q1 = infer.query(['Starts'], evidence={'Radio': 0, 'Fuel': 0})
    print("\nP(Starts | Radio, Fuel):\n", q1)

    # P(Battery | Moves=true)
    q2 = infer.query(['Battery'], evidence={'Moves': 0})
    print("\nP(Battery | Moves):\n", q2)

    # P(Radio | Starts=false)
    q3 = infer.query(['Radio'], evidence={'Starts': 1})
    print("\nP(Radio | ~Starts):\n", q3)


if __name__ == "__main__":
    main()