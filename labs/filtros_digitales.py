import numpy as np
from scipy import signal


def design_fir_lowpass(fs, cutoff_hz, numtaps=101, window='hamming'):
    nyq = fs / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError("cutoff_hz debe estar entre 0 y Nyquist (fs/2)")
    h = signal.firwin(numtaps, cutoff_hz/nyq, window=window)
    return h


def design_iir_butter_lowpass(fs, cutoff_hz, order=4):
    nyq = fs / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError("cutoff_hz debe estar entre 0 y Nyquist (fs/2)")
    b, a = signal.butter(order, cutoff_hz/nyq, btype='low', analog=False)
    return b, a


def freq_response_from_taps(b, a=1, worN=1024, fs=1.0):
    w, h = signal.freqz(b, a, worN=worN)
    # w está en rad/muestra, convertir a Hz
    f = w * fs / (2 * np.pi)
    return f, h


def apply_fir(h, x, axis=-1):
    return signal.lfilter(h, 1.0, x, axis=axis)


def apply_iir(b, a, x, zero_phase=False, axis=-1):
    if zero_phase:
        return signal.filtfilt(b, a, x, axis=axis)
    else:
        return signal.lfilter(b, a, x, axis=axis)


def plot_response_matplotlib(axs, f, H, title=None):
    import matplotlib.pyplot as plt
    ax_mag, ax_phase = axs
    ax_mag.plot(f, 20*np.log10(np.maximum(np.abs(H), 1e-12)))
    ax_mag.set_ylabel('Magnitud (dB)')
    ax_mag.set_xlabel('Frecuencia (Hz)')
    if title:
        ax_mag.set_title(title)
    ax_phase.plot(f, np.angle(H))
    ax_phase.set_ylabel('Fase (rad)')
    ax_phase.set_xlabel('Frecuencia (Hz)')


if __name__ == '__main__':
    fs = 44100
    h = design_fir_lowpass(fs, 3000, numtaps=101)
    b, a = design_iir_butter_lowpass(fs, 3000, order=4)
    f_fir, H_fir = freq_response_from_taps(h, 1, fs=fs)
    f_iir, H_iir = freq_response_from_taps(b, a, fs=fs)
    print('Módulo de filtros cargado. Coeficientes FIR:', len(h), 'IIR b len:', len(b))
