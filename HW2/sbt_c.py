#
# sbt.py
# Sideband Thermometry, this file holds the implementation for part 3 of HW 2
# Author: Frederic zur Bonsen
# E-Mail: fzurbonsen@ethz.ch
# 

import argparse, csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.optimize import curve_fit
import cmath as cm


# function to load the CSV file
def load_csv(file):
    with open(file, 'r', newline='') as f:
        reader = list(csv.reader(f))

    freqs = []
    for x in reader[0]:
        x = x.replace("+0i", "")
        freqs.append(float(x))

    psd_ = []
    for x in reader[1]:
        x = x.replace("i", "j")
        psd_.append(complex(x))

    f = np.array(freqs, dtype=float)
    psd = np.array(psd_, dtype=complex)

    return f, psd


# function to plot a single trace
def plot_1_trace(trace):
    plt.figure(figsize=(8, 4))
    plt.plot(trace)
    plt.xlabel("Time steps")
    plt.ylabel("Power")
    plt.title("Single Time Trace")
    # plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


# funcion to plot 1 PSD
def plot_1_psd(f, psd_trace):
    plt.figure()
    plt.semilogy(f, psd_trace)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.title("Power Spectral Density")
    plt.grid(True)
    plt.show()
    return


# function to plot motional sidebands
def plot_zoomed_mosb(window, lo_freq, f, psd_trace, y_max=None, y_min=None):

    # prepare data
    mask = (f > lo_freq - window) & (f < lo_freq + window)
    f_shifted = (f[mask] - lo_freq )* 1e-3
    psd_norm = psd_trace[mask] / np.min(psd_trace[mask])

    plt.figure(figsize=(8,4))
    # plt.semilogy(f_shifted, psd_norm)
    plt.plot(f_shifted, psd_norm)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("PSD")
    plt.title("Zoomed-in PSD around LO")
    if y_max is not None:
        plt.ylim(top=y_max)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    plt.grid(True)
    plt.show()


# function to plot window
def plot_window(f, psd_trace, y_max=None, y_min=None):

    # prepare data
    f_shifted = f * 1e-3
    psd_norm = psd_trace / np.min(psd_trace)

    plt.figure(figsize=(8,4))
    # plt.semilogy(f_shifted, psd_norm)
    plt.plot(f_shifted, psd_norm)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("PSD")
    plt.title("Zoomed-in PSD around LO")
    if y_max is not None:
        plt.ylim(top=y_max)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    plt.grid(True)
    plt.show()
    return


# function to plot window
def plot_window_curve_fit(f, psd_trace, curve, y_max=None, y_min=None):

    # prepare data
    f_shifted = f * 1e-3
    psd_norm = psd_trace / np.min(psd_trace)
    curve_norm = curve / np.min(psd_trace)

    plt.figure(figsize=(8,4))
    # plt.semilogy(f_shifted, psd_norm)
    plt.plot(f_shifted, psd_norm, label='Data')
    plt.plot(f_shifted, curve_norm, 'r-', label='Fit', linewidth=2)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("PSD")
    plt.title("Zoomed-in PSD around LO")
    if y_max is not None:
        plt.ylim(top=y_max)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    plt.grid(True)
    plt.savefig("img/ex3_fit.eps")
    plt.show()
    return


# function to calculate the power spectral density
def calculate_psd(trace, fs):
    trace = np.asarray(trace)
    f, Pxx = welch(trace, fs=fs, window='hann', nperseg=min(256, len(trace)))
    return f, Pxx


# function to calculate the power spectral density
def calculate_psd_neg(trace, fs):
    trace = np.asarray(trace)
    N = len(trace)
    X = np.fft.fft(trace * np.hanning(N))
    freqs = np.fft.fftfreq(N, d=1/fs)
    PSD = (np.abs(X)**2) / N
    return freqs, PSD


# function of a lorentzian
def lorentzian(f, A, f0, gamma, B):
    # f0: center frequency
    # gamma: linewidth
    # A: amplitude
    # B: offset
    return A * (gamma**2) / ((f - f0)**2 + gamma**2) + B


# function to calculate the phonon population
def calc_phonon_population(freqs, psd_real, popt):
    N = []
    for psd, freq in zip(psd_real, freqs):
        n = 0.5*(psd / lorentzian(freq, *popt) - 1)
        N.append(n)

    return np.mean(N)


# main function
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, required=True,
                        help="Trace file [.csv]")
    args = parser.parse_args()

    # parameters
    f_m = 53e3

    f, psd = load_csv(args.file)

    # Exercise c)
    psd_real = np.array(psd).real

    # fit lorentzian
    popt, pcov = curve_fit(lorentzian, f, psd_real, p0=[1, f_m, 1e3, 0])
    
    # plot the fitted curve on the data
    # fit_curve = lorentzian(f, *popt)
    fit_curve = lorentzian(f, *popt)
    plot_window_curve_fit(f, psd_real, fit_curve)

    n_phonon = calc_phonon_population(f, psd_real, popt)
    print(f"Estimated phonon population: n = {n_phonon:.2f}")
    return 0



if __name__ == "__main__":
    main()