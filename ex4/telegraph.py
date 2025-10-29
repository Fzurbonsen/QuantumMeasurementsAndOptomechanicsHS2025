# 
# telegraph.py:
# This file holds the implementation of the exercise 4 from QMO.
# Author: Frederic zur Bonsen
# E-Mail: <fzurbonsen@ethz.ch>
# 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# function to load the .csv file and create 
def load_csv_data(path):
    df = pd.read_csv(path, header=None)
    input_data = []
    for i in range(0, 20):
        input_data.append(df[i])
    return input_data


# function to plot a siganl
def plot_signal(signal):
    plt.plot(signal)

    # add labels and title
    plt.xlabel("time steps")
    plt.ylabel("signal")

    plt.show()


# function to plot psd
def plot_psd(freqs, psd_avg):
    plt.plot(freqs, psd_avg)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD (A^2/Hz)")
    plt.grid(True)
    plt.show()


# function to plot fit
def plot_fit(freqs, psd_avg, popt):
    plt.loglog(freqs, psd_avg, label="Measured PSD", lw=1.5)
    plt.loglog(freqs, psd_model(freqs, *popt), label="Fitted Model", lw=2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD (A²/Hz)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.show()


# claculate PSD
def calc_PSD(signals, sampling_frequency):
    N = len(signals[0])
    psd_list = []

    # iterate over all singals
    for signal in signals:
        fft_signal = np.fft.fft(signal) # compute FFT
        psd = (1/(sampling_frequency*N)) * np.abs(fft_signal)**2 # take the absoulute of the fft_signal and scale to Hz
        psd_list.append(psd[:N//2])

    psd_avg = np.mean(psd_list, axis=0) # take the mean of all the signals
    psd_avg[1:-1] *= 2 # adjust to keep the power

    freqs = np.fft.fftfreq(N, d=1/sampling_frequency)[:N//2] # calculate the freqeuncy scale

    return freqs, psd_avg


# function to model the PSD
def psd_model(f, a, R, noise):
    return (4 * a **2 * R) / ((2 * np.pi * f)**2 + (2 * R)**2) + noise # we model the PSD with the calculated PSD from the exercise + some noise


# function to fit the PSD
def fit_psd(freqs, psd_avg):
    f = freqs[1:] # skip 0
    P = psd_avg[1:] # skip 0

    popt, pcov = curve_fit(psd_model, f, P, p0=[1, 1e3, 1e-14]) # use scipy curve_fit to fit the curve to our PSD model
    a_fit, R_fit, noise_fit = popt
    print(f"Fit parameters:\n  a = {a_fit:.3e}\n  R = {R_fit:.3e} Hz\n  noise = {noise_fit:.3e}")
    return popt


# function to check Parseval's theorem
def check_parseval(input_data, freqs, psd_avg):
    variance = np.var(np.concatenate(input_data))
    psd_integral = np.sum(psd_avg)*(freqs[1]-freqs[0])
    print(f"Variance: {variance:.3f}, Integral of PSD: {psd_integral:.3f}")


# main funtion
def main():
    input_data = load_csv_data("./data/dataNoise.csv") # load data
    freqs, psd_avg = calc_PSD(input_data, 1e6) # calcualte the PSD of the data
    plot_psd(freqs, psd_avg) # plot the average PSD
    check_parseval(input_data, freqs, psd_avg)
    
    popt = fit_psd(freqs, psd_avg) # fit the PSD model to the PSD data
    plot_fit(freqs, psd_avg, popt) # plot the fitted model

    


if __name__ == "__main__":
    main()