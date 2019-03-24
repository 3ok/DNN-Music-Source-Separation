import numpy as np 
from scipy.signal import stft, istft
import dsdtools
from constants import *
from math import pi

def extract_stft(track):
    """Extract amplitude and phase from an audio track
    
    Arguments:
        track {Track} -- input audio track
    
    Returns:
        phase {array} -- phase of the STFT : (nb_bins, nb_channels, nb_time_windows)
        amplitude {array} -- amplitude of the STFT : (nb_bins, nb_channels, nb_time_windows)
    """

    _, _, track_stft = stft(track.audio, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_OVERLAP)
    phase, amplitude = np.angle(track_stft), np.absolute(track_stft)

    return amplitude, phase

def time_diff(phase):
    """Returns time differences of the phase
    
    Arguments:
        phase {array} -- phase of a signal : (nb_bins, nb_channels, nb_time_windows)
    
    Returns:
        dt_phase {array} -- time differences of phase : (nb_bins, nb_channels, nb_time_windows)
    """

    dt_phase = np.diff(phase)
    # zero-padding at the beginning to have the same size as phase
    dt_phase = np.concatenate([np.zeros((phase.shape[0], phase.shape[1], 1)), dt_phase], axis=2)
    dt_phase = (dt_phase - pi) % (2*pi) - pi

    return dt_phase

def frequency_diff(phase):
    """Returns frequency differences of the phase
    
    Arguments:
        phase {array} -- phase of a signal : (nb_bins, nb_channels, nb_time_windows)
    
    Returns:
        df_phase {array} -- frequency differences of phase : (nb_bins, nb_channels, nb_time_windows)
    """

    df_phase = np.diff(phase, axis=0)
    # zero-padding at the beginning to have the same size as phase
    df_phase = np.concatenate([np.zeros((1, phase.shape[1], phase.shape[2])), df_phase], axis=0)
    df_phase = (df_phase - pi) % (2*pi) - pi

    return df_phase

def time_correct(phase):
    """Returns time differences of the phase after correction by compensation
    
    Arguments:
        phase {array} -- phase of a signal : (nb_bins, nb_channels, nb_time_windows)
    
    Returns:
        dt_phase {array} -- time differences of phase after correction : (nb_bins, nb_channels, nb_time_windows)
    """

    dt_phase = time_diff(phase)
    for k in range(dt_phase.shape[0]):
        dt_phase -= 2*pi*k*(HOP_SIZE / N_FFT)
    dt_phase = (dt_phase - pi) % (2*pi) - pi

    return dt_phase

def frequency_correct(phase):
    """Returns frequency differences of the phase after correction by compensation
    
    Arguments:
        phase {array} -- phase of a signal : (nb_bins, nb_channels, nb_time_windows)
    
    Returns:
        dt_phase {array} -- frequency differences of phase after correction : (nb_bins, nb_channels, nb_time_windows)
    """

    df_phase = frequency_diff(phase) + pi
    df_phase = (df_phase - pi) % (2*pi) - pi

    return df_phase

def extract_context(idx, a, context_size = CONTEXT_SIZE):
    """Extract a context of an array around a certain index
    
    Arguments:
        idx {int} -- central index of the context
        a {array} -- array from which we want to extract context : (nb_bins, nb_channels, nb_time_windows)
    
    Keyword Arguments:
        context_size {int} -- size of the context (default: {CONTEXT_SIZE})
    
    Returns:
        context {array} -- context around index (nb_bins, nb_channels, 2*context_size + 1)
    """
    if idx < context_size:
        # we zero-pad the beginning
        context = np.concatenate([
            np.zeros((a.shape[:2] + (context_size - idx,))),
            a[:,:, :idx + context_size + 1]
        ], axis=-1)
    
    elif idx + context_size < a.shape[-1]:
        context = a[:,:,idx - context_size:idx + context_size + 1]

    else:
        context = np.concatenate([
            a[:,:,idx - context_size:],
            np.zeros((a.shape[:2] + (idx + context_size - a.shape[-1] + 1),))
        ], axis=-1)

    return context

def process_track(track):
    """Process a track to obtain corrected spectral features
    
    Arguments:
        track {Track} -- track from which we want to extract features
    
    Returns:
        amplitude {array} -- amplitude of the STFT
        dt_phase {array} -- corrected phase time differences
        df_phase {array} -- corrected phase frequency differentiaces
    """

    amplitude, phase = extract_stft(track)
    dt_phase, df_phase = time_correct(phaese), frequency_correct(phase)

    return amplitude, dt_phase, df_phase

def process_all(tracks, context_size=CONTEXT_SIZE):
    """Processes a list of tracks in order to prepare them to be fed to the network
    
    Arguments:
        tracks {list} -- list of tracks
    
    Keyword Arguments:
        context_size {int} -- size of the context (default: {CONTEXT_SIZE})
    
    Returns:
        amplitudes {list} -- list of amplitudes within context : (nb_bins, nb_channels, 2*context_size + 1)
        phases {list} -- list of concatenated phase differences within context : (nb_bins, 2*nb_channels, 2*context_size + 1)
    """

    amplitudes, phases = [], []
    for track in tracks:
        amplitude, dt_phase, df_phase = process_track(track)
        # inserting features to lists
        amplitudes += [extract_context(i, amplitude, context_size=CONTEXT_SIZE) for i in range(amplitude.shape[-1])]
        phases += [np.concatenate([
            extract_context(i, dt_phase, context_size=CONTEXT_SIZE),
            extract_context(i, df_phase, context_size=CONTEXT_SIZE)
        ], axis=1)
        for i in range(df_phase.shape[-1])]

    return amplitudes, phases

def reconstruct(amplitude, phase):
    """Reconstructs audio track from amplitude and phase
    
    Arguments:
        amplitude {array} -- amplitude of the STFT
        phase {array} -- phase of the STFT
    
    Returns:
        track_audio {array} -- reconstucted audio track
    """

    track_audio = istft(amplitude * np.exp(1j * np.phase), fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_OVERLAP, freq_axis=0)
    return track_audio