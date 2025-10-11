import numpy as np
import pandas as pd
import math as ma
import matplotlib.pyplot as plt


def quantum_jump_simulator(e_state_0, g_state_0, g):

    _theta = lambda g, delta_t: g*delta_t

    detect_right_prob = lambda theta, e_state, g_state: 0.5*(1 + ma.sin(theta))*e_state**2 + 0.5*(1 - ma.sin(theta))*g_state**2
    detect_left_prob = lambda theta, e_state, g_state: 0.5*(1 - ma.sin(theta))*e_state**2 + 0.5*(1 + ma.sin(theta))*g_state**2

    prob_right_tracker = []
    prob_left_tracker = []

    for t in range(100):
        prob_right_tracker.append(detect_right_prob(_theta(g, t/5), e_state_0, g_state_0))
        prob_left_tracker.append(detect_left_prob(_theta(g, t/5), e_state_0, g_state_0))

    df = pd.DataFrame({
        'time': range(100),
        'prob_right': prob_right_tracker,
        'prob_left': prob_left_tracker
    })

    df.plot(x='time', y=['prob_right', 'prob_left'], title='Quantum Jump Simulation')
    plt.xlabel('Cycle')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()


def main():
    # quantum_jump_simulator(ma.sqrt(1), ma.sqrt(0), 0.5*ma.pi)

    g = 0.5
    e_prob = 1
    g_prob = 0

    print("Starting Simulator:")
    while (True):
        print("<sim>:")
        print("<sim>:   Enter a command.")
        user_input = input("<sim>:   ")

        if user_input == 'quit' or user_input ==  'q':
            break

        elif user_input == 'run':
            print("<sim>:")
            print("<sim>:   Running simulation.")
            quantum_jump_simulator(ma.sqrt(e_prob), ma.sqrt(g_prob), g*ma.pi)
            print("<sim>:   Finished simulating.")
        
        elif user_input == 'set':
            print("<sim>:")
            g = float(input("<sim>:   g = "))
            e_prob = float(input("<sim>:   e_prob = "))
            g_prob = float(input("<sim>:   g_prob = "))

        elif user_input == 'help':
            print("<sim>:")
            print("<sim>:   quantum_jump_simulator:")
            print("<sim>:   help:       help menu")
            print("<sim>:   run:        run the simulation")
            print("<sim>:   quit/q:     quit the simulator")

        else:
            print("<sim>:")
            print("Type --help for help")
    


if __name__ == "__main__":
    main()
