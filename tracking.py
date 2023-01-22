import numpy as np
from tensorflow import keras
from scipy import interpolate
import utils

def get_koopman_VDP(mu, tau=1, time_delay = 55, time_delay_spacing = 1, rank = 16):
    """Approximate the global linearization of a Van der Pol oscillator.

    The simulation follows the differential equation
        dx/dt = y/tau
        dy/dt = (mu * (1 - x**2) * y - x)/tau

    The global linearization is approximated using dynamic mode decomposition to
    construct a rank 16 matrix using 55 timesteps as delay embeddings.

    Args:
        mu: The intrinsic parameter of a Van der Pol oscillator
        tau: Timescaling factor for the Van der Pol oscillator
        time_delay: Number of delay embeddings
        time_delay_spacing: Number of timesteps between delay embeddings
        rank: Rank of the DMD matrix

    Returns:
        An approximation of the Koopman operator through Dynamic Mode Decomposition
    """
    dmd = utils.DMD(truncation='hard',
                    threshold=rank,
                    time_delay=time_delay,
                    time_delay_spacing=time_delay_spacing)

    x0  = [1.0, 3.0]
    period_length = 11.45015*8
    num_periods_simulate = 15
    sampling_rate_simulate = 512

    timestep_count = int(num_periods_simulate*sampling_rate_simulate)
    dt_simulate = num_periods_simulate*period_length/timestep_count

    vdp_simulation = utils.simulate_vanderpol_oscillator(   dt_simulate,
                                                            timestep_count, 
                                                            x0=x0, 
                                                            mu=mu, 
                                                            tau=tau)[0]

    vdp_solution = vdp_simulation[0:1,:]
    training_data = (vdp_solution[:,1:])
    dmd.fit(training_data, dt_simulate)
    return dmd.A


def hankel_matrix(Xin, n_delay_coordinates, spacing=1):
    """Form the Hankel matrix of a timeseries.

    Args:
        Xin: A multivariate timeseries, where each column represents a timestep
        n_delay_coordinates: Number of delay coordinates
        spacing: Timesteps between delay cooordinates.

    Returns:
        A Hankel matrix constructed from the set of timeseries
    """
    n_inputs, n_samples = Xin.shape

    X = np.zeros((n_inputs * (n_delay_coordinates), n_samples - spacing*(n_delay_coordinates-1)))
    for i in range(n_delay_coordinates):
        idxs = np.arange(spacing*i, spacing*(i+1) + n_samples - spacing*n_delay_coordinates)
        X[i*n_inputs:(i+1)*n_inputs] = Xin[:, idxs]
    return X


def define_autoencoder_structure(time_delay, dim=6):
    """Returns the autoencoder used in the manuscript

    Args:
        time_delay: Number of delay embeddings in the Koopman operators

    Returns:
        A tuple (encoder, decoder, autoencoder)
        where encoder and decoder are the two components of the keras autoencoder.
    """

    # Define Encoder Structure
    encoder_input = keras.Input(shape=(time_delay, time_delay, 1), name='img')
    x = keras.layers.Flatten()(encoder_input)
    x_0 = keras.layers.Dense(1000, activation="relu")(x)
    x_1 = keras.layers.Dense(500, activation="relu")(x_0)
    x_2 = keras.layers.Dense(100, activation="elu")(x_1)
    x_3 = keras.layers.Dense(40, activation="elu")(x_2)
    encoder_output = keras.layers.Dense(dim, activation="elu")(x_3)
    encoder = keras.Model(encoder_input, encoder_output, name='encoder')

    # Define Decoder Structure
    decoder_input = keras.layers.Dense(dim, activation="elu")(encoder_output)
    y_0 = keras.layers.Dense(40, activation="elu")(decoder_input)
    y_1 = keras.layers.Dense(100, activation="elu")(y_0)
    y_2 = keras.layers.Dense(500, activation="relu")(y_1)
    y_3 = keras.layers.Dense(1000, activation="relu")(y_2)
    y_4 = keras.layers.Dense(time_delay**2)(y_3)
    decoder_output = keras.layers.Reshape((time_delay, time_delay, 1))(y_4)
    decoder = keras.Model(encoder_output, decoder_output, name='decoder')

    # Assemble Autoencoder
    autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')

    return (encoder, decoder, autoencoder)


def train_autoencoder(autoencoder, Atilde):
    """Trains the autoencoder in the manuscript

    Args:
        autoencoder: The autoencoder to be trained
        Atilde: The set of approximate Koopman operators
    """
    opt = keras.optimizers.legacy.Adam(lr=0.0005, decay=1e-6)
    autoencoder.compile(opt, loss="mse")
    autoencoder.fit(Atilde,
                    Atilde,
                    epochs=500,
                    batch_size=32, 
                    validation_split=0.10)


def generate_cubic_curve(encoder, Atilde):
    """Generates a cubic curve of systems in the latent space
    of the autoencoder.

    Args:
        encoder: Encoding portion of a trained autoencoder
        Atilde: Dataset of linear systems from SLDS model

    Returns:
        A tuple containing (encode_train, encode_interpolated) where
    encode_train is Atilde pushed through the encoder and
    encode_interpolated is a set of points from cubic curve fit through encode_train.

    Notes:
        6D latent dimension is hardcoded
    """
    N = Atilde.shape[0]
    time_delay = Atilde.shape[1]
    dim = 6

    # Pass the training data through the encoder
    encode_train = np.zeros([N,dim])
    for i in range(Atilde.shape[0]):
        input_sample = Atilde[i,:,:].reshape(-1, time_delay, time_delay, 1)
        encode_train[i,:] = encoder.predict(input_sample, verbose=0)[0]

    # Construct the interpolators
    label = np.linspace(0,1,N)
    f1 = interpolate.interp1d(label, encode_train[:,0], kind = 'cubic')
    f2 = interpolate.interp1d(label, encode_train[:,1], kind = 'cubic')
    f3 = interpolate.interp1d(label, encode_train[:,2], kind = 'cubic')
    f4 = interpolate.interp1d(label, encode_train[:,3], kind = 'cubic')
    f5 = interpolate.interp1d(label, encode_train[:,4], kind = 'cubic')
    f6 = interpolate.interp1d(label, encode_train[:,5], kind = 'cubic')

    # Evaluate the interpolators
    labels = np.linspace(0,1,20000)
    encode_interpolated = np.zeros([20000,dim])
    for i in range(20000):
        encode_interpolated[i,0] = f1(labels[i])
        encode_interpolated[i,1] = f2(labels[i])
        encode_interpolated[i,2] = f3(labels[i])
        encode_interpolated[i,3] = f4(labels[i])
        encode_interpolated[i,4] = f5(labels[i])
        encode_interpolated[i,5] = f6(labels[i])

    return encode_train, encode_interpolated



def simulate_switching_system(mu_vals, win_fixed, sig_y):
    """Simulate a switching Van der Pol system

    Args:
        mu_vals: values of the parameter in the system
        win_fixed: Switching interval
        sig_y: parameter step variance

    Returns:
        A tuple (timeseries, koopman_operators) where timeseries is the
        state of the system and koopman_operators is the set of
        linearizations of the dynamical system at each timestep. 
    """

    # Set parameters
    x0  = [1.0, 2.0]
    index_0 = 25
    mu_0 = mu_vals[index_0]
    T = 1500
    tau = 1

    period_length = 11.45015*8
    num_periods_simulate = 15
    sampling_rate_simulate = 512

    timestep_count = int(num_periods_simulate*sampling_rate_simulate)
    dt_simulate = num_periods_simulate*period_length/timestep_count

    A = np.zeros((55,55))

    # Initialize the simulation arrays
    mu_s = [mu_0]
    timeseries = []
    koopman_operators = []

    # Simulate the systems parameter evolution
    for t in range(T):
        # Evolve the parameter at the given switching interval
        if t%win_fixed==0:
            mu_s.append(mu_s[-1]+sig_y*np.random.randn(1)[0])
            A = get_koopman_VDP(mu_s[-1])
        else:
            mu_s.append(mu_s[-1])

        # Simulate the system and get the state matrix
        simulation = utils.simulate_vanderpol_oscillator(dt_simulate, 2,
                                                   x0=x0, mu=mu_s[-1], tau=tau)[0]
        # Get the second value of the state
        timeseries.append(simulation[0,1])
        x0 = simulation[:,1]
        koopman_operators.append(A)

    # Convert the list to a Numpy array
    timeseries = np.array(timeseries)
    koopman_operators = np.array(koopman_operators)

    return timeseries, koopman_operators



def construct_cubic_interpolation(Atilde):
    """Construct a cubic interpolation Atilde"""
    label = np.linspace(0,1,Atilde.shape[0])

    f = [interpolate.interp1d(label, Atilde[:,i], kind = 'cubic') for i in range(6)]
    interpolation = lambda x : [f[i](x) for i in range(6)]

    return label, interpolation



def maximum_likelihood_windowed_detection_point(timeseries, timeindex, Atilde, window_params):
    """Detect system in a window based on maximum likelihood

    Args:
        timeseries: The full dataset
        timeindex: Starting index of the window
        Atilde: Candidate linear dynamical systems

    Returns:
        The index of the best fit linear system according to ML detection.
    """

    # Set the parameters
    time_delay, delay_keep, T, index_0 = window_params
    N = 50

    # Get overlapping sequences offset by 1
    state_previous = []
    state_current = []
    for k in range(delay_keep):
        state_previous.append((timeseries[timeindex-time_delay-k-1:timeindex-k -1]))
        state_current.append(timeseries[timeindex-time_delay-k:timeindex-k])
    
    # Convert to arrays
    state_previous = np.array(state_previous).T
    state_current = np.array(state_current).T
    
    min_error = np.linalg.norm(state_current - Atilde[0,:,:]@state_previous)
    best_index = 0
    for j in range(N):
        current_error = np.linalg.norm(state_current - Atilde[j,:,:]@state_previous)
        if current_error<min_error:
            min_error = current_error
            best_index = j

    return best_index
    


def prior_windowed_detection_point(timeseries, timeindex, previous_mode, Atilde):
    """Detect system in a window based on the usage of a prior

    Args:
        timeseries: The full dataset
        timeindex: Starting index of the window
        previous_mode: Index of the previous mode
        Atilde: Candidate linear dynamical systems

    Returns:
        The index of the best fit linear system based on nearest neighbor transitions.
    """

    # Set the parameters
    time_delay = 55
    delay_keep = 50
    N = 50
    search_radius = 2

    # Get overlapping sequences offset by 1
    state_previous = []
    state_current = []
    for k in range(delay_keep):
        state_previous.append((timeseries[timeindex-time_delay-k-1:timeindex-k -1]))
        state_current.append(timeseries[timeindex-time_delay-k:timeindex-k])
    
    # Convert to arrays
    state_previous = np.array(state_previous).T
    state_current = np.array(state_current).T
    
    min_error = np.linalg.norm(state_current - Atilde[0,:,:]@state_previous)
    best_index = 0
    for j in range(max(0,previous_mode - search_radius), min(N, previous_mode + search_radius)):
        current_error = np.linalg.norm(state_current - Atilde[j,:,:]@state_previous)
        if current_error<min_error:
            min_error = current_error
            best_index = j

    return best_index
    

def latent_windowed_detection_point(timeseries, timeindex, current_latent, interpolation, embedded_train, decoder):
    """Detect system in a window based on the latent space tracking

    Args:
        timeseries: The full dataset
        timeindex: Starting index of the window
        current_latent: current latent space point
        interpolation: interpolation function for Atilde
        decoder: Autoencoder decoder

    Returns:
        The index of the best fit linear system based on nearest neighbor transitions.
    """

    # Set the parameters
    time_delay = 55
    delay_keep = 100
    search_radius = 2
    dim = 6
    search_radius = 0.02
    neighbor_count = 21

    state_previous = []
    state_current = []
    for k in range(delay_keep):
        state_previous.append((timeseries[timeindex-time_delay-k-1:timeindex-k-1]))
        state_current.append(timeseries[timeindex-time_delay-k:timeindex-k])
    
    # Convert to arrays
    state_previous = np.array(state_previous).T
    state_current = np.array(state_current).T


    neighborhood = list(np.linspace(-2*search_radius+current_latent, 2*search_radius+current_latent, neighbor_count))

    lower, center, upper = quantize_and_get_neighbors(current_latent, 0.02)
    neighborhood.append(lower)
    neighborhood.append(center)
    neighborhood.append(upper)

    neighborhood = np.array(neighborhood)

    minimum_error = 1e10

    # This doesn't seem to check the last three elements
    for j in range(neighbor_count):

        latent_point = np.array(interpolation(neighborhood[j]))
        latent_point = latent_point.reshape(1,dim)

        decoded_operator = decoder(latent_point).numpy().reshape(time_delay,time_delay)

        current_error = np.linalg.norm(state_current - decoded_operator@state_previous)
        if current_error<minimum_error:
            minimum_error = current_error
            A_estimate = decoded_operator
            current_latent = neighborhood[j]


    return A_estimate, current_latent



def prior_windowed_detection(timeseries, Atilde, window_params):
    """Estimate the time-varying Koopman operator based on the prior

    Args:
        timeseries: The state of the system over time
        Atilde: The set of possible models in the SLDS
        window_params: A tuple of parameters for the windows

    Returns:
        A numpy array of the estimated matrix for each timestep
    """
    time_delay, delay_keep, T, index_0 = window_params

    # Setting parameters
    previous_mode = index_0

    learned_A_prior = []
    for i in range(time_delay+delay_keep,T):
        best_index = prior_windowed_detection_point(timeseries, i, previous_mode, Atilde)
        previous_mode = best_index
        learned_A_prior.append(Atilde[best_index,:,:])

    return np.array(learned_A_prior)



def latent_windowed_detection(timeseries, embedded_train, decoder, window_params):
    """Estimate the time-varying Koopman operator by tracking the latent space

    Args:
        timeseries: The state of the system over time
        embedded_train: The set of possible models in the SLDS in the latent space
        decoder: The decoder which converts from the latent space to Koopman operators
        window_params: A tuple of parameters for the windows

    Returns:
        A numpy array of the estimated matrix for each timestep
    """

    time_delay, delay_keep, T, index_0 = window_params

    # Construct the interpolator
    label, interpolation = construct_cubic_interpolation(embedded_train)

    # Setup variables
    current_mode = index_0
    current_latent = label[current_mode]
    learned_A_local_search = []

    for i in range(time_delay+delay_keep,T):
        A_estimate, current_latent = latent_windowed_detection_point(   timeseries,
                                                                        i,
                                                                        current_latent,
                                                                        interpolation,
                                                                        embedded_train,
                                                                        decoder)
        learned_A_local_search.append(A_estimate)

    return np.array(learned_A_local_search)



def maximum_likelihood_windowed_detection(timeseries, Atilde,  window_params):
    """Estimate the time-varying Koopman operator with a switching linear dynamical system

    Args:
        timeseries: The state of the system over time
        Atilde: The set of possible models in the SLDS
        window_params: A tuple of parameters for the windows

    Returns:
        A numpy array of the estimated matrix for each timestep
    """

    # Unpack the parameters
    time_delay, delay_keep, T, index_0 = window_params

    # Initialize Arrays
    learned_A_sw = []

    # Construct the sequence of windowed ML detections
    for i in range(time_delay+delay_keep,T):
        best_index = maximum_likelihood_windowed_detection_point(timeseries, i, Atilde, window_params)
        learned_A_sw.append(Atilde[best_index,:,:])

    return np.array(learned_A_sw)



def quantize_and_get_neighbors(point, stepsize):
    """Quantize the value and get neighboring nodes

    Args:
        point: Center value to quantize
        stepsize: Quantization step sizes

    Returns:
        A tuple (lower, center, upper) where center is the quantization
        floor of point, lower is the value stepsize below, and upper is the
        value stepsize above.
    """

    upper = stepsize * ((point + stepsize)//stepsize)
    center = stepsize * (point // stepsize)
    lower = stepsize * ((point - stepsize)//stepsize)

    return lower, center, upper


