import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils

from tracking import get_koopman_VDP
from tracking import define_autoencoder_structure
from tracking import train_autoencoder
from tracking import generate_cubic_curve
from tracking import simulate_switching_system
from tracking import maximum_likelihood_windowed_detection
from tracking import latent_windowed_detection
from tracking import prior_windowed_detection


def plot_koopman_distance_heatmap(distances):
    """Generates the heatmap of distances between Koopman operators
    based on the set of Dynamic Mode Decomposition matrices.

    Figure 1 (b) in
    'A Continuous Representation of Switching Linear Dynamic Systems for Accurate Tracking',
    by Parisa Karimi, Helmuth Naumer, and Farzad Kamalabadi

    Args:
        distances: Matrix of distances between Koopman operators
    """

    plt.figure(figsize = (7,7))
    plt.imshow(distances, origin = 'lower')
    plt.colorbar()
    plt.xlabel('index of parameter $\mu$')
    plt.ylabel('index of parameter $\mu$')
    plt.title('Distance between data points')
    plt.savefig('fig1/dists_all.pdf')



def plot_koopman_distance_slice(distances):
    """Generates a slice of distances between Koopman operators
    based on the set of Dynamic Mode Decomposition matrices.

    Figure 1 (a) in
    'A Continuous Representation of Switching Linear Dynamic Systems for Accurate Tracking',
    by Parisa Karimi, Helmuth Naumer, and Farzad Kamalabadi

    Args:
        distances: Matrix of distances between Koopman operators
    """

    plt.figure(figsize = (8/1.2,5/1.2))
    plt.plot(distances[0])
    plt.axhline(4.1,ls='--')
    plt.axvline(22,ls='--', color = 'red')
    plt.axvline(30,ls='--', color = 'red')
    plt.xlabel("index of parameter $\mu$")
    plt.ylabel("distance")
    plt.grid()
    plt.savefig('fig1/dists_2.pdf',bbox_inches='tight')



def plot_koopman_eigen_similarity(similarities, distances, rank):
    """Generates a plot of similarities between eigenvectors
    and eigenvalues for two different modes of the Koopman operator

    Args:
        similarities: Matrix of similarities between eigenvectors
        distances: Matrix of distances between eigenvalues
        rank: DMD rank, used for scaling
    """
    plt.figure(figsize = (12/1.6,8/1.6))
    k = 0
    plt.plot(similarities[k,0:]/rank)
    plt.plot(distances[k,0:])
    plt.legend(['eigenvector similarity', 'eigenvalue distance'])
    plt.grid()
    plt.xlabel("index of parameter $\mu$")
    plt.axvline(21, color = 'red',  ls = '--')
    plt.axhline(0.18959504629079826, color = 'blue', alpha = .7, ls = '--')
    plt.axvline(30, color = 'red', ls = '--')
    plt.savefig("fig1/simanddis.pdf",bbox_inches='tight')



def koopman_distance_plots():
    """Generates Figures 1 (a-c) in the manuscript.
    Files are saved in the `fig1` directory.
    """
    # Configure matplotlib
    font = {'family' : 'normal',
            'size'   : 18}
    matplotlib.rc('font', **font)

    # Set the parameters used in the manuscript
    N = 50
    time_delay = 55
    time_delay_spacing = 1
    rank = 16
    dim = 6

    # Initialize the range of mu values to be tested
    mu_vals = np.linspace(3, 7, N)

    # Compute approximate Koopman Operators
    Atilde = np.zeros([N,time_delay,time_delay])
    for count_x, mu in enumerate(mu_vals):
        Atilde[count_x, :, :] = get_koopman_VDP(mu, 1, time_delay, time_delay_spacing, rank)

    # Compute distances between the operators
    distances = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            distances[i][j] = (np.linalg.norm(Atilde[i,:,:]-Atilde[j,:,:]))

    # Plot Figures (a) and (b)
    plot_koopman_distance_heatmap(distances)
    plot_koopman_distance_slice(distances)

    # Compute the similarity matrices
    eigen_distances = np.zeros([N,N])
    similarities = np.zeros([N,N])

    for i in range(N):
        sig0,U0 = np.linalg.eig(Atilde[i,:,:])

        for ki in range(50):
            if U0[0,ki]<0:
                U0[:,ki] = -U0[:,ki]

        U0 = np.linalg.inv(U0)
        for j in range(N):
            sig1,U1 = np.linalg.eig(Atilde[j,:,:])
            for ki in range(50):
                if U1[0,ki]<0:
                    U1[:,ki] = -U1[:,ki]
            U1 = np.linalg.inv(U1)
            tot = 0
            for k in range(rank):
                dum1 = np.conjugate((U0[k,:])).T @ (U1[k,:])
                tot += abs(dum1)
            eigen_distances[i,j] = np.linalg.norm(sig0 - sig1)
            similarities[i,j] = tot

    # Plot Figure (c)
    plot_koopman_eigen_similarity(similarities, eigen_distances, rank)

    # Construct and train the autoencoder
    encoder, decoder, autoencoder = define_autoencoder_structure(time_delay, dim)
    train_autoencoder(autoencoder, Atilde)

    # Generate the manifold curve data
    encode_train, encode_interpolated = generate_cubic_curve(encoder, Atilde)

    # Plot figure (d)
    encoder_plots(encode_train, encode_interpolated)

    return Atilde, mu_vals, decoder, encode_train



def encoder_plots(encode_train, encode_interpolated):
    """Generate the encoder curve plot from the final data

    Args:
        encode_train: Input samples for the encoder
        encode_interpolated: Fitted points from the interpolation

    """
    ax = plt.axes(projection ='3d')

    ax.figure.set_size_inches(8/1.2, 8/1.2)

    ax.scatter(encode_train[:,0],
               encode_train[:,1],
               encode_train[:,2],
               marker ='x',
               color = 'b',
               alpha = .8,
               label = 'data points')

    ax.scatter(encode_interpolated[:,0],
               encode_interpolated[:,1],
               encode_interpolated[:,2],
               marker ='o',
               color = 'r',
               alpha = .01,
               label = 'fitted points')

    plt.grid()
    plt.savefig("fig1/curve.pdf")
    plt.show()



def switching_montecarlo_plot(error_table, error_ntable, error_nei, win_fixed, time_delay, delay_keep):
    """Generate the Monte Carlo plot

    Args:
        error_table: SLDS operator error
        error_ntable: SLDS operator error with known prior
        error_nei: neighborhood search operator error
        Win_fixed:
        time_delay:
        delay_keep:
    """

    plt.figure(figsize = (12/1.2,8/1.2))
    plt.plot(error_table, 'r x', alpha = .4, label='switching system' )
    plt.plot(error_ntable, 'b--', alpha = .9, label = 'switching system with prior')
    plt.plot(error_nei,'c*-',  alpha = .7, label = 'neighborhood search')
    plt.legend(loc = 'upper left')
    plt.xlabel('time')
    plt.ylabel('dynamic operator error')
    plt.grid()
    for i in range(2,10):
        line_position = i*Win_fixed-(time_delay+delay_keep)
        plt.axvline(line_position, ls = '--', alpha = .5,color = 'black')
        plt.savefig('fig2/MC_nei_search_final.pdf')
        plt.show()


def compute_tracking_error(Atilde, embedded_train, mu_vals, decoder):
    """Computes the tracking error for three competing techniques
    applied to a Van der Pol oscillator with a time-varying parameter.

    Args:
        Atilde:
        mu_vals:
        decoder:

    Returns:
        A tuple (error_A_switching, error_A_prior, error_A_local_search) corresponding
        to the Koopman operator estimation error under the three algorithms
    """

    # Set the parameters
    index_0 = 25
    win_fixed = 100
    sig_y = .02
    T = 1000
    time_delay = 55
    delay_keep = 100
    window_params = time_delay, delay_keep, T, index_0

    ### Simulate the System
    timeseries, koopman_operators = simulate_switching_system(mu_vals, win_fixed, sig_y)


    # Evaluate three competing methods
    learned_A_switching     = maximum_likelihood_windowed_detection(timeseries, Atilde, window_params)
    learned_A_prior         = prior_windowed_detection(timeseries, Atilde,  window_params)
    learned_A_local_search  = latent_windowed_detection(timeseries, embedded_train, decoder, window_params)

    ### Compute errors
    error_A_switching = []
    error_A_prior = []
    error_A_local_search = []

    for i in range(time_delay+delay_keep,T):
        estimate_index = i-time_delay-time_delay

        ground_truth       = koopman_operators[i, :, :]
        estimate_switching = learned_A_switching[estimate_index,:,:]
        estimate_prior     = learned_A_prior[estimate_index,:,:]
        estimate_local     = learned_A_local_search[estimate_index,:,:]

        error_A_switching.append(np.linalg.norm(estimate_switching - ground_truth))
        error_A_prior.append(np.linalg.norm(estimate_prior - ground_truth))
        error_A_local_search.append(np.linalg.norm(estimate_local - ground_truth))


    return error_A_switching, error_A_prior, error_A_local_search



def montecarlo_tracking_error(Atilde, embedded_train, mu_vals, decoder):
    """Monte carlo simulation of the tracking error for three competing
       techniques applied to the Van der Pol oscillator with a 
       time-varying parameter.

    Args:
        Atilde:
        mu_vals:
        decoder:

    Returns:
        A tuple (error_table, error_ntable, error_nei) corresponding
        to the Koopman operator estimation error under the three algorithms.
    """

    repeat = 100
    T = 1000
    delay_keep = 100
    time_delay = 55

    error_table = np.zeros(T-(time_delay+delay_keep))
    error_ntable = np.zeros(T-(time_delay+delay_keep))
    error_nei = np.zeros(T-(time_delay+delay_keep))

    for rep in range(repeat):
        error_A_switching, error_A_prior, error_A_local_search = compute_tracking_error(Atilde, embedded_train, mu_vals, decoder)
        error_table += np.array(error_A_switching).reshape(-1)
        error_ntable += np.array(error_A_prior).reshape(-1)
        error_nei += np.array(error_A_local_search).reshape(-1)

    return error_table/repeat, error_ntable/repeat, error_nei/repeat
