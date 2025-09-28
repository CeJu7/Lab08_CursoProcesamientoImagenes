import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from typing import Tuple, Optional, List, Union
import warnings

plt.style.use(
    'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')


class TransformadaFourier:
    def __init__(self, fs: float = 1.0):
        self.fs = fs
        self.dt = 1.0 / fs

    def dft(self, x: np.ndarray) -> np.ndarray:
        N = len(x)
        X = np.zeros(N, dtype=complex)

        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

        return X

    def idft(self, X: np.ndarray) -> np.ndarray:
        N = len(X)
        x = np.zeros(N, dtype=complex)

        for n in range(N):
            for k in range(N):
                x[n] += X[k] * np.exp(2j * np.pi * k * n / N)

        return x / N

    def fft_analysis(self, x: np.ndarray,
                     ventana: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(x)

        if ventana is None:
            x_windowed = x
        else:
            if ventana.lower() in ['hann', 'hanning']:
                w = signal.get_window('hann', N)
            elif ventana.lower() == 'hamming':
                w = signal.get_window('hamming', N)
            elif ventana.lower() == 'blackman':
                w = signal.get_window('blackman', N)
            elif ventana.lower() == 'bartlett':
                w = signal.get_window('bartlett', N)
            else:
                print(f"Advertencia: Ventana '{ventana}' no reconocida. Usando ventana rectangular.")
                w = np.ones(N)

            x_windowed = x * w

        X = fft(x_windowed)
        freqs = fftfreq(N, self.dt)

        magnitud = np.abs(X)
        fase = np.angle(X)

        return freqs, magnitud, fase

    def espectrograma(self, x: np.ndarray,
                      nperseg: int = 256,
                      noverlap: Optional[int] = None,
                      ventana: str = 'hanning') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if noverlap is None:
            noverlap = nperseg // 2

        f, t, Sxx = signal.spectrogram(x, fs=self.fs,
                                       window=ventana,
                                       nperseg=nperseg,
                                       noverlap=noverlap)

        return f, t, Sxx

    def densidad_espectral_potencia(self, x: np.ndarray,
                                    nperseg: int = 256,
                                    ventana: str = 'hanning') -> Tuple[np.ndarray, np.ndarray]:
        f, Pxx = signal.welch(x, fs=self.fs, window=ventana, nperseg=nperseg)
        return f, Pxx

    def filtro_paso_bajo(self, X: np.ndarray,
                         freqs: np.ndarray,
                         fc: float) -> np.ndarray:
        X_filtered = X.copy()
        X_filtered[np.abs(freqs) > fc] = 0
        return X_filtered

    def filtro_paso_alto(self, X: np.ndarray,
                         freqs: np.ndarray,
                         fc: float) -> np.ndarray:
        X_filtered = X.copy()
        X_filtered[np.abs(freqs) < fc] = 0
        return X_filtered

    def filtro_paso_banda(self, X: np.ndarray,
                          freqs: np.ndarray,
                          fc_low: float,
                          fc_high: float) -> np.ndarray:
        X_filtered = X.copy()
        mask = (np.abs(freqs) < fc_low) | (np.abs(freqs) > fc_high)
        X_filtered[mask] = 0
        return X_filtered

    def convolucion_fft(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        N = len(x) + len(h) - 1

        x_padded = np.zeros(N)
        h_padded = np.zeros(N)

        x_padded[:len(x)] = x
        h_padded[:len(h)] = h

        X = fft(x_padded)
        H = fft(h_padded)
        Y = X * H
        y = np.real(ifft(Y))

        return y

    def correlacion_fft(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        N = len(x) + len(y) - 1

        x_padded = np.zeros(N)
        y_padded = np.zeros(N)

        x_padded[:len(x)] = x
        y_padded[:len(y)] = y

        X = fft(x_padded)
        Y = fft(y_padded)
        Rxy = X * np.conj(Y)
        rxy = np.real(ifft(Rxy))

        return rxy


def generar_señales_test(fs: float = 1000, duracion: float = 1.0) -> dict:
    t = np.arange(0, duracion, 1/fs)

    señales = {
        'senoidal': np.sin(2 * np.pi * 50 * t),
        'multitono': np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t),
        'chirp': signal.chirp(t, f0=10, f1=100, t1=duracion),
        'ruido_blanco': np.random.randn(len(t)),
        'pulso': signal.square(2 * np.pi * 5 * t),
        'triangular': signal.sawtooth(2 * np.pi * 10 * t, width=0.5),
        'exponencial': np.exp(-t) * np.sin(2 * np.pi * 25 * t),
    }

    return t, señales


def calcular_snr(señal: np.ndarray, ruido: np.ndarray) -> float:
    potencia_señal = np.mean(señal**2)
    potencia_ruido = np.mean(ruido**2)

    if potencia_ruido == 0:
        return float('inf')

    snr_db = 10 * np.log10(potencia_señal / potencia_ruido)
    return snr_db


def aplicar_ruido(señal: np.ndarray, snr_db: float) -> np.ndarray:
    potencia_señal = np.mean(señal**2)
    potencia_ruido = potencia_señal / (10**(snr_db/10))
    ruido = np.sqrt(potencia_ruido) * np.random.randn(len(señal))

    return señal + ruido


def plot_espectro(freqs: np.ndarray,
                  magnitud: np.ndarray,
                  fase: Optional[np.ndarray] = None,
                  titulo: str = "Espectro de Frecuencia",
                  log_scale: bool = False) -> None:
    if fase is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))

    if log_scale:
        ax1.semilogy(freqs, magnitud)
        ax1.set_ylabel('Magnitud (log)')
    else:
        ax1.plot(freqs, magnitud)
        ax1.set_ylabel('Magnitud')

    ax1.set_title(titulo)
    ax1.grid(True, alpha=0.3)

    if fase is not None:
        ax2.plot(freqs, np.degrees(fase))
        ax2.set_xlabel('Frecuencia (Hz)')
        ax2.set_ylabel('Fase (grados)')
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel('Frecuencia (Hz)')

    plt.tight_layout()
    plt.show()


def plot_espectrograma(f: np.ndarray,
                       t: np.ndarray,
                       Sxx: np.ndarray,
                       titulo: str = "Espectrograma") -> None:
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
    plt.ylabel('Frecuencia (Hz)')
    plt.xlabel('Tiempo (s)')
    plt.title(titulo)
    plt.colorbar(label='Potencia (dB)')
    plt.tight_layout()
    plt.show()
