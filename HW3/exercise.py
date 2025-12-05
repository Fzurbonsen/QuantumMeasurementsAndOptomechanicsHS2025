'''
exercise.py
This file implements the solutions to HW3 of the Quantum Measurements and
Optomechanics lecture tought by Prof. Dr. Martin Frimmer.
Author: Frederic zur Bonsen
E-Mail: fzurbonsen@ethz.ch
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
*****************
* Global Config *
*****************
'''

# system parameters
f_cutoff = 4 # cutoff frequency = 4Hz
trace_len = 1e6 # time traces of length 10^6
f_sample = 10 # sample rate = 10Hz
n_bar = 1 # thermal occupatiopn due to the thermal batch coupling = 1
Gamma_qb = 1e-1 # measurement backaction rate = 0.1Hz
eta_d = 0.6 # quantum efficiency = 0.6
Gamma = 1e-2 # oscillator damping rate = 0.01Hz
Gamma_meas = eta_d * Gamma_qb # measurement rate
Vc_bar = (n_bar + 0.5) + (Gamma_qb/Gamma) # steady state of the conditional state variance
dt = 1 / f_sample # discrete time step

# data paramters
raw_data_file = "data/ix_csv_50x1mio.txt"
raw_data = None
raw_data = pd.read_csv(raw_data_file, header=None)

mean_PSD_data_file = "data/mean_psd.csv"
mean_PSD_data = None
if os.path.exists(mean_PSD_data_file):
  mean_PSD_data = pd.read_csv(mean_PSD_data_file)

predictive_filter_data_file = "data/predictive_filter.csv"
predictive_filter_data = None
if os.path.exists(predictive_filter_data_file):
  predictive_filter_data = pd.read_csv(predictive_filter_data_file)

predictive_filter_mean_PSD_data_file = "data/predictive_filter_mean_psd.csv"
predictive_filter_mean_PSD_data = None
if os.path.exists(predictive_filter_mean_PSD_data_file):
  predictive_filter_mean_PSD_data = pd.read_csv(predictive_filter_mean_PSD_data_file)



'''
********************
* Helper Functions *
********************
'''

# function to get X_bar (quadrature) from a time trace (i_x(t))
def get_X_bar(i_x):
  calibration_factor = 2*np.sqrt(Gamma_meas)
  X_bar = np.array(i_x/calibration_factor)
  return X_bar


# function to compute the power spectral density
def compute_PSD(X, fs):
  N = len(X)
  Xf = np.fft.rfft(X) # one sided FFT
  PSD = (np.abs(Xf)**2) / (N * fs) # scale PSD
  f = np.fft.rfftfreq(N, 1/fs) # build frequency axis
  return PSD, f


# function to compute the predictive filtering on a single trace
def predictive_filter(i_x):
  N = len(i_x)
  X = np.zeros(N, dtype=np.float64)

  '''
  We intizialize at 0. Therefore X[0] = 0
  '''
  for k in range(N-1):
    dW = i_x[k] * dt - 2 * np.sqrt(Gamma_meas) * X[k] * dt
    X[k+1] = X[k] - (Gamma/2) * X[k] * dt + (2 * np.sqrt(Gamma_meas) * Vc_bar) * dW
  return X


# function to compute power spectral filter |H(f)|^2
def power_spectral_filter(f):
  omega = 2*np.pi*f
  H = (4*Gamma_meas*Vc_bar**2) / ((Gamma/2 + 4*Gamma_meas*Vc_bar)**2 + omega**2)
  return H



'''
*************
* Exercises *
*************
'''

'''
Exercise f):
Let us first look at the raw, unfiltered position measurement. 
To this end, as an example, plot X̄ which is an appropriately 
scaled version of i_x from the first time trace of the dataset 
in the time window between 46 × 10^3 and 47 × 10^3 s.
What is the calibration factor?
'''

# function implementing exercise f)
def exercise_f():
  print("exercise f)")
  '''
  This function assumes that the raw_data DataFrame is populated.
  '''

  i_x = raw_data.iloc[0].values # get the first time trace
  t_window_start = 46e3 # starting time of the window
  t_window_end = 47e3 # end time of the window

  # adjust for sampling rate
  window_start = int(t_window_start * f_sample)
  window_end = int(t_window_end * f_sample)

  # construct window
  i_x_window = i_x[window_start:window_end]

  # convert to X_bar
  X_bar = get_X_bar(i_x_window)

  # build time axis
  t = np.arange(window_start, window_end) / f_sample / 1e3

  # plot
  plt.plot(t, X_bar)
  plt.xlabel("time [10^3 s]")
  plt.ylabel("X̄")
  plt.grid(True)
  plt.savefig("img/exercise_f.eps")
  plt.savefig("img/exercise_f.png", dpi=300)
  plt.show()

  print(f"The calibration factor is 2*sqrt(Gamma_meas) = {2*np.sqrt(Gamma_meas)}")
  return

'''
The calibration factor is 2*sqrt(Gamma_meas) = 0.4898979485566356
'''


'''
Exercise g):
Look at the measured, unfiltered time traces X̄ in the spectral 
domain. Plot the power spectral density of X̄, averaged over all 
50 experimental repetitions.
Use a double logarithmic plot and discuss salient features.
'''

# function implementing exercise g)
def exercise_g():
  print("exercise g)")
  global mean_PSD_data
  '''
  This function assumes that the raw_data DataFrame is populated.
  '''

  '''
  This part of the code only has to be executed once to produce the mean PSD file.
  '''
  if mean_PSD_data is None:
    PSD_list = [] # initialize empty list to hold PSDs

    # iterate over all rows of the data frame
    for idx in range(len(raw_data)):
      i_x = raw_data.iloc[idx].values # get time trace
      X_bar = get_X_bar(i_x) # convert to X_bar
      PSD, f = compute_PSD(X_bar, f_sample) # compute PSD and frequency scale
      PSD_list.append(PSD) # add PSD to the list

    # compute the mean of the PSDs
    PSD_array = np.array(PSD_list)
    mean_PSD = PSD_array.mean(axis=0)

    # convert to DataFrame
    df_mean_PSD = pd.DataFrame({
      "Frequency": f,
      "Mean_PSD": mean_PSD
    })

    # save to .csv
    df_mean_PSD.to_csv(mean_PSD_data_file, index=False)

    mean_PSD_data = df_mean_PSD

  '''
  Continue with the mean_PSD_data.
  '''
  # plot the mean psd
  plt.loglog(mean_PSD_data["Frequency"], mean_PSD_data["Mean_PSD"])
  plt.xlabel("Frequency [Hz]")
  plt.ylabel("PSD")
  plt.grid(True)
  plt.savefig("img/exercise_g.eps")
  plt.savefig("img/exercise_g.png", dpi=300)
  plt.show()
  return


'''
Exercise h):
Carry out the predictive filtering according to Eqs. (2) and (3)
to determine −→X from the measurement record for each of the 50 
experimental runs provided. Comment on how you treat the very
beginning of the time trace. Show your result for −→X in the time
window from Problem (f) and compare to the unfiltered data X̄.
Describe what you observe.
'''

# function implementing exercise h)
def exercise_h():
  print("exercise h)")
  '''
  This function assumes that the raw_data DataFrame is populated.
  '''
  global predictive_filter_data
  
  '''
  This part of the code only has to be executed once to produce
  the predictive filter file.
  '''
  if predictive_filter_data is None:
    predictive_filter_list = [] # initialize empty list to hold predictive filter

    # iterate over all rows of the data frame
    for idx in range(len(raw_data)):
      i_x = raw_data.iloc[idx].values # get time trace
      X_arrow = predictive_filter(i_x)
      predictive_filter_list.append(X_arrow)

    # convert to DataFram
    predictive_filter_df = pd.DataFrame(predictive_filter_list)

    # save to .csv
    predictive_filter_df.to_csv(predictive_filter_data_file, index=False)

    predictive_filter_data = predictive_filter_df

  # plot
  X_arrow = predictive_filter_data.iloc[0].values # get the predictive filter of the first time trace
  t_window_start = 46e3 # starting time of the window
  t_window_end = 47e3 # end time of the window

  # adjust for sampling rate
  window_start = int(t_window_start * f_sample)
  window_end = int(t_window_end * f_sample)

  # construct window
  X_arrow_window = X_arrow[window_start:window_end]

  # build time axis
  t = np.arange(window_start, window_end) / f_sample / 1e3

  # plot
  plt.plot(t, X_arrow_window)
  plt.xlabel("time [10^3 s]")
  plt.ylabel("X_arrow")
  plt.grid(True)
  plt.savefig("img/exercise_h.eps")
  plt.savefig("img/exercise_h.png", dpi=300)
  plt.show()
  return


'''
Exercise i):
What is the variance of your predicted position (quadrature) <−→X^2>? 
Average all experimental data. What is the result you expect? Where 
may the difference be from?
'''

# function implementing exercise i)
def exercise_i():
  print("exercise i)")
  '''
  This function assumes that predictive_filter_data is populated.
  '''

  # compute the variance
  trace_variances = predictive_filter_data.var(axis=1, ddof=0)

  # compute the mean variance accross all traces
  mean_variance = trace_variances.mean()

  print(f"Mean variance of the predictive filter (⟨X^2⟩): {mean_variance}")
  return

'''
Mean variance = 17.687109413264082
'''


'''
Exercise j):
Compare the variance of your conditional estimates <−→X^2> with the
conditional state variance Vc and the unconditional variance Vuc.
Which relation do you expect? Does the data match your expectation?
'''

# function implementing exercise j)
def exercise_j():
  print("exercise j)")
  '''
  Nothing to be done
  '''
  return


'''
Exercise k):
isualize the filtering process in the spectral domain. Plot the
power spectrum of −→X and, for comparison, that of X̄ from Problem (g).
'''

# function implementing exercise k)
def exercise_k():
  print("exercise k)")
  global predictive_filter_mean_PSD_data
  '''
  This function assumes that predictive_filter_data is populated
  '''
  PSD_list = []

  if predictive_filter_mean_PSD_data is None:
    for idx in range(len(predictive_filter_data)):
      X_arrow = predictive_filter_data.iloc[idx].values
      PSD, f = compute_PSD(X_arrow, f_sample)
      PSD_list.append(PSD)

    # compute the mean of the PSDs
    PSD_array = np.array(PSD_list)
    mean_PSD = PSD_array.mean(axis=0)

    # convert to DataFrame
    df_mean_PSD = pd.DataFrame({
      "Frequency": f,
      "Mean_PSD": mean_PSD
    })

    df_mean_PSD.to_csv(predictive_filter_mean_PSD_data_file)

    predictive_filter_mean_PSD_data = df_mean_PSD

  '''
  Continue with the mean_PSD_data.
  '''
  # plot the mean psd
  plt.loglog(predictive_filter_mean_PSD_data["Frequency"], predictive_filter_mean_PSD_data["Mean_PSD"])
  plt.xlabel("Frequency [Hz]")
  plt.ylabel("PSD")
  plt.grid(True)
  plt.savefig("img/exercise_k.eps")
  plt.savefig("img/exercise_k.png", dpi=300)
  plt.show()
  return


'''
Exercise l):
Derive the spectral filtering function corresponding to the temporal
filter defined by Eq. (2) and add it to your plot from Problem (k).
'''

# function implementing exercise l)
def exercise_l():
  print("exercise l)")
  global predictive_filter_mean_PSD_data
  '''
  This function assumes that predictive_filter_mean_PSD_data is populated
  '''
  f = predictive_filter_mean_PSD_data["Frequency"].values
  H = []

  for freq in f:
    H_ = power_spectral_filter(freq)
    H.append(H_)

  H = np.array(H)

  plt.loglog(f, predictive_filter_mean_PSD_data["Mean_PSD"], label="Mean PSD")
  plt.loglog(f, H, label="Filter |H(f)|²")
  plt.xlabel("Frequency [Hz]")
  plt.ylabel("PSD")
  plt.legend()
  plt.grid(True)
  plt.savefig("img/exercise_l.eps")
  plt.savefig("img/exercise_l.png", dpi=300)
  plt.show()
  return


'''
*****************
* Main Function *
*****************
'''

def main():
  exercise_f()
  exercise_g()
  exercise_h()
  exercise_i()
  exercise_j()
  exercise_k()
  exercise_l()
  return




if __name__ == "__main__":
  main()