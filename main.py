#!/bin/python

from figures import koopman_distance_plots
from figures import montecarlo_tracking_error
from figures import switching_montecarlo_plot


Atilde, mu_vals, decoder, encode_train = koopman_distance_plots()
#error_A_switching, error_A_prior, error_A_local_search = montecarlo_tracking_error(Atilde, mu_vals, decoder)


win_fixed = 100
time_delay = 55
delay_keep = 50

error_table, error_ntable, error_nei = montecarlo_tracking_error(Atilde, encode_train, mu_vals, decoder)
switching_montecarlo_plot(error_table, error_ntable, error_nei, win_fixed, time_delay, delay_keep)
