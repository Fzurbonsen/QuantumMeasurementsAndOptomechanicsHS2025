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


# function to load the CSV file
def load_csv(file):
    traces = []
    with open(file, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            traces.append([float(x) for x in row])
    return traces


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

    f_shifted = f * 1e-3
    psd_norm = psd_trace / np.min(psd_trace)

    plt.figure()
    plt.semilogy(f_shifted, psd_norm)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("PSD / min(PSD)")
    plt.grid(True)
    plt.savefig("img/psd.eps")
    plt.show()
    return


# function to plot all PSDs
def plot_all_psd(f_avg, psd_traces, psd_avg):
    plt.figure()
    for psd_ in psd_traces:
        plt.semilogy(f_avg, psd_, color='gray', alpha=0.3)
    plt.semilogy(f_avg, psd_avg, color='red', label='Average')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.grid(True)
    plt.legend()
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
    plt.ylabel("PSD / min(PSD)")
    if y_max is not None:
        plt.ylim(top=y_max)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    plt.grid(True)
    plt.savefig("img/zoomed_psd.eps")
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
    plt.ylabel("PSD / min(PSD)")
    if y_max is not None:
        plt.ylim(top=y_max)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    plt.grid(True)
    plt.savefig("img/curve_fit_psd.eps")
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



# function to average all the PSDs
def average_psd(freqs, psd_traces):
    psd_array = np.array(psd_traces)
    psd_avg = np.mean(psd_array, axis=0)
    f = freqs[0]
    return f, psd_avg


# function of a lorentzian
def lorentzian(f, A, f0, gamma, B):
    # f0: center frequency
    # gamma: linewidth
    # A: amplitude
    # B: offset 
    return A * (gamma**2) / ((f0 - f)**2 + gamma**2) + B

fixed_gamma = 0
def fixed_lorentzian(f, A, f0, B):
    return lorentzian(f, A, f0, fixed_gamma, B)


# main function
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, required=True,
                        help="Trace file [.csv]")
    args = parser.parse_args()

    # sampling frequency
    f_sample = 3.75e6
    f_lo = 980e3
    f_m = 53e3
    window_lo = 1.3*f_m

    traces = load_csv(args.file)
    # plot_1_trace(traces[0][0:1000])


    # Exercise a)
    f = []
    psd_traces = []
    for trace in traces:
        # f_, psd_ = calculate_psd(trace, f_sample)
        f_ , psd_ = calculate_psd_neg(trace, f_sample)
        f.append(f_)
        psd_traces.append(psd_)
    f_avg, psd_avg = average_psd(f, psd_traces)
    
    plot_1_psd(f_avg, psd_avg)
    # plot_all_psd(f_avg, psd_traces, psd_avg)

    plot_zoomed_mosb(window_lo, f_lo, f_avg, psd_avg, 5.5, 1)


    # Exercise b)
    window_sb = f_m * 0.1
    global fixed_gamma

    # prepare data
    mask_sb1 = (f_avg > f_lo + f_m - window_sb) & (f_avg < f_lo + f_m + window_sb)
    data_sb1 = psd_avg[mask_sb1]
    f_sb1 = f_avg[mask_sb1]
    mask_sb2 = (f_avg > f_lo - f_m - window_sb) & (f_avg < f_lo - f_m + window_sb)
    data_sb2 = psd_avg[mask_sb2]
    f_sb2 = f_avg[mask_sb2]

    # plot windows around sidebands
    # plot_window(f_sb1, data_sb1, 5.5, 1)
    # plot_window(f_sb2, data_sb2, 5.5, 1)

    # fit the lorentzian to the data
    # popt1, pcov1 = curve_fit(lorentzian, f_sb1, data_sb1, p0=[1, f_lo + f_m, 1e3, 0])
    # popt2, pcov2 = curve_fit(lorentzian, f_sb2, data_sb2, p0=[1, f_lo - f_m, 1e3, 0])
    popt1, pcov1 = curve_fit(lorentzian, f_sb1, data_sb1, p0=[1, f_lo + f_m, 1e3, 0])
    # popt2, pcov2 = curve_fit(lorentzian, f_sb2, data_sb2, p0=[1, f_lo - f_m, 1e3, 0])
    fixed_gamma = popt1[2]
    popt2, pcov2 = curve_fit(fixed_lorentzian, f_sb2, data_sb2, p0=[1, f_lo - f_m, 0])

    print(popt1)
    print(popt2)

    # plot the fitted curve on the data
    # fit_curve1 = lorentzian(f_sb1, *popt1)
    # fit_curve2 = lorentzian(f_sb2, *popt2)
    fit_curve1 = lorentzian(f_sb1, *popt1)
    fit_curve2 = fixed_lorentzian(f_sb2, *popt2)
    plot_window_curve_fit(f_sb1, data_sb1, fit_curve1, 5.5, 1)
    plot_window_curve_fit(f_sb2, data_sb2, fit_curve2, 5.5, 1)

    

    # calculate phonon number
    A_upper = popt2[0]
    A_lower = popt1[0]
    n_phonon = A_lower / (A_upper - A_lower)
    print(f"Estimated phonon population via amplitude: n = {n_phonon:.2f}")
    Area_upper = np.trapz(fit_curve2, f_sb2)
    Area_lower = np.trapz(fit_curve1, f_sb1)
    n_phonon = Area_lower / (Area_upper - Area_lower)
    print(f"Estimated phonon population via area: n = {n_phonon:.2f}")
    return 0



if __name__ == "__main__":
    main()