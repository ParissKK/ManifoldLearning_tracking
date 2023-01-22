#!/bin/python

from figures import koopman_distance_plots
from figures import montecarlo_tracking_error
from figures import switching_tracking_plot
from figures import compute_tracking_error

win_fixed = 100
time_delay = 55
delay_keep = 100

print("Generating Figure 1...")

Atilde, mu_vals, decoder, encode_train = koopman_distance_plots()

print("Generating Single Sample Simulation")
error_A_switching, error_A_prior, error_A_local_search = compute_tracking_error(Atilde, encode_train, mu_vals, decoder)

print("Plotting the Single Sample Simulation")
switching_tracking_plot(error_A_switching, error_A_prior, error_A_local_search, win_fixed, time_delay, delay_keep, "singleSample.pdf")

print("Running Monte Carlo Simulations...")


error_table, error_ntable, error_nei = montecarlo_tracking_error(Atilde, encode_train, mu_vals, decoder)
print("Plotting Figure 2 (b)...")
switching_tracking_plot(error_table, error_ntable, error_nei, win_fixed, time_delay, delay_keep, "MC_nei_search_final.pdf")
