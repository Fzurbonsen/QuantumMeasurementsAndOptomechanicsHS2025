#
# qunatim_jumps_sim.py:
# This file holds the implementation of a simple two state system with a Rabi drive that is measured with a continuous ancilla stream.
# This simulation was written within the scope of the Quantum Measurements and Optomechanics lecture by Martin Frimmer at ETHZ.
# Author: Frederic zur Bonsen
# E-Mail: <fzurbonsen@ethz.ch>
#

import numpy as np
import pandas as pd
import cmath as ma
import matplotlib.pyplot as plt


def quantum_jump_simulator(e_state_0, g_state_0, g, omega, delta_t, tmax):

    _theta = lambda g, delta_t: g*delta_t
    _rho = lambda omega, delta_t: omega*delta_t

    _c_2_norm = lambda a: a * a.conjugate()

    _detect_right_prob = lambda theta, e_state, g_state: 0.5*(1 + ma.sin(theta))*_c_2_norm(e_state) + 0.5*(1 - ma.sin(theta))*_c_2_norm(g_state)
    _detect_left_prob = lambda theta, e_state, g_state: 0.5*(1 - ma.sin(theta))*_c_2_norm(e_state) + 0.5*(1 + ma.sin(theta))*_c_2_norm(g_state)

    _update_state_e_right = lambda theta, rho, e_state, g_state: 0.5*((ma.exp(-1j*theta) - 1j)*e_state - 1j*rho*(ma.exp(1j*theta) - 1j)*g_state)
    _update_state_g_right = lambda theta, rho, e_state, g_state: 0.5*((ma.exp(1j*theta) - 1j)*g_state - 1j*rho*(ma.exp(-1j*theta) - 1j)*e_state)

    _update_state_e_left = lambda theta, rho, e_state, g_state: 0.5*((ma.exp(-1j*theta) + 1j)*e_state - 1j*rho*(ma.exp(1j*theta) + 1j)*g_state)
    _update_state_g_left = lambda theta, rho, e_state, g_state: 0.5*((ma.exp(1j*theta) + 1j)*g_state - 1j*rho*(ma.exp(-1j*theta) + 1j)*e_state)

    _N = lambda e_state, g_state: 1/ma.sqrt(_c_2_norm(e_state) + _c_2_norm(g_state)) # this computes the normalization constant of a function |phi> = e_state|e> + g_state|g>

    # initialize atom state trackers
    prob_e_state = _c_2_norm(e_state_0).real
    prob_g_state = _c_2_norm(g_state_0).real
    prob_e_tracker = [prob_e_state]
    prob_g_tracker = [prob_g_state]

    #initialize ancilla tracker
    anciall_sum = 0
    ancilla_tracker = []

    # set initial state for the atom
    e_state = e_state_0 
    g_state = g_state_0

    # calculate interaction phase and rabi phase
    theta = _theta(g, delta_t)
    rho = _rho(omega, delta_t)
    
    # calculate normalization factor for P(+1) and P(-1) i.e. the ensuring that the post ancilla detection states sum to one
    N = 1/ma.sqrt(1 + rho**2)

    for t in range(tmax):

        prob_right = _detect_right_prob(_theta(g, delta_t), e_state, g_state).real
        prob_left = _detect_left_prob(_theta(g, delta_t), e_state, g_state).real

        # decide in what state we detect the ancilla
        random = np.random.choice([1, -1], p=[prob_right, prob_left])

        # update the state of the atom
        if random == 1:     # ancilla detected in |->y> state (right state)
            e_state = _update_state_e_right(theta, rho, e_state, g_state)
            g_state = _update_state_g_right(theta, rho, e_state, g_state)
            # we now need to normalize the wave function e_state|e> + g_state|g>
            N_right = _N(e_state, g_state)
            e_state *= N_right
            g_state *= N_right

        elif random == -1:  # ancilla detected in |<-y> state (left state)
            e_state = _update_state_e_left(theta, rho, e_state, g_state)
            g_state = _update_state_g_left(theta, rho, e_state, g_state)
            # we now need to normalize the wave function e_state|e> + g_state|g>
            N_left = _N(e_state, g_state)
            e_state *= N_left
            g_state *= N_left

        # get the state probabilities after the ancilla interaction
        prob_e_state = _c_2_norm(e_state).real
        prob_g_state = _c_2_norm(g_state).real

        # update trackers
        prob_e_tracker.append(prob_e_state)
        prob_g_tracker.append(prob_g_state)
        anciall_sum += random
        ancilla_tracker.append(anciall_sum)

    # create figure with 2 subplots (stacked vertically)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)  # 2 rows, 1 column

    # plot states
    df_states = pd.DataFrame({
        'iteration': range(tmax+1),
        'exicted state': prob_e_tracker,
        'ground state': prob_g_tracker
    })
    df_states.plot(x='iteration', y=['exicted state', 'ground state'], ax=axes[0], title='Atom State Probabilities')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Probability')
    axes[0].grid(True)

    # plot ancilla walk
    df_ancilla = pd.DataFrame({
        'iteration': range(tmax),
        'ancilla walk': ancilla_tracker,
    })
    df_ancilla.plot(x='iteration', y='ancilla walk', ax=axes[1], title='Ancilla Walk')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Walk Sum')
    axes[1].grid(True)

    plt.tight_layout()  # adjust spacing
    # plt.show()
    plt.savefig('./plot.png')

    plt.close()


def main():

    g = 0.1
    omega = 0.001
    delta_t = 0.5
    e_prob = 0.5
    g_prob = 0.5
    tmax = 10000

    # welcome message
    print("Starting Simulator:")
    print("<sim>:   config:")
    print(f"<sim>:   g = {g}\t\t# interaction strength frequency")
    print(f"<sim>:   omega = {omega}\t\t# rabi drive frequency")
    print(f"<sim>:   delta_t = {delta_t}\t\t# interaction time")
    print(f"<sim>:   e_prob = {e_prob}\t\t# probability of being in exited state at t=0")
    print(f"<sim>:   g_prob = {g_prob}\t\t# probability of being in ground state at t=0")
    print(f"<sim>:   tmax = {tmax}\t\t# number of ancillas the atom interacts with")
    print("<sim>:")

    while (True):
        print("<sim>:")
        user_input = input("<sim>:   ")

        if user_input == 'quit' or user_input ==  'q':
            break

        elif user_input == 'run' or user_input == 'r':
            print("<sim>:")
            print("<sim>:   Running simulation.")
            quantum_jump_simulator(ma.sqrt(e_prob), ma.sqrt(g_prob), g*ma.pi, omega*ma.pi, delta_t, tmax)
            print("<sim>:   Finished simulating.")
        
        elif user_input == 'set' or user_input == 's':
            print("<sim>:")
            g = float(input("<sim>:   g = "))
            omega = float(input("<sim>:   omega = "))
            delta_t = float(input("<sim>:   delta_t = "))
            e_prob = float(input("<sim>:   e_prob = "))
            g_prob = float(input("<sim>:   g_prob = "))
            tmax = int(input("<sim>:   tmax = "))

        elif user_input == 'config' or user_input == 'c':
            print("<sim>:")
            print(f"<sim>:   g = {g}\t\t# interaction strength frequency")
            print(f"<sim>:   omega = {omega}\t\t# rabi drive frequency")
            print(f"<sim>:   delta_t = {delta_t}\t\t# interaction time")
            print(f"<sim>:   e_prob = {e_prob}\t\t# probability of being in exited state at t=0")
            print(f"<sim>:   g_prob = {g_prob}\t\t# probability of being in ground state at t=0")
            print(f"<sim>:   tmax = {tmax}\t\t# number of ancillas the atom interacts with")

        elif user_input == 'help' or user_input == 'h':
            print("<sim>:")
            print("<sim>:   quantum_jump_simulator:")
            print("<sim>:   help/h:     help menu")
            print("<sim>:   run/r:      run the simulation")
            print("<sim>:   set/s:      set the config")
            print("<sim>:   config/c:   show the config")
            print("<sim>:   quit/q:     quit the simulator")

        else:
            print("<sim>:")
            print("<sim>:   Type 'help' for help")
    


if __name__ == "__main__":
    main()
