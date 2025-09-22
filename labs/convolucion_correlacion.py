from scipy import signal
import soundfile as sf
from typing import Sequence
import numpy as np


def linear_convolution(x: Sequence, h: Sequence) -> np.ndarray:
    """Computa la convolución lineal discreta x * h (directa).

    Args:
        x: secuencia de entrada
        h: respuesta/otro kernel

    Returns:
        Resultado de la convolución como numpy array.
    """
    x = np.asarray(x)
    h = np.asarray(h)
    nx = x.size
    nh = h.size
    y = np.zeros(nx + nh - 1, dtype=np.result_type(x, h))
    for n in range(y.size):
        k_min = max(0, n - (nh - 1))
        k_max = min(n, nx - 1)
        s = 0
        for k in range(k_min, k_max + 1):
            s += x[k] * h[n - k]
        y[n] = s
    return y


def conv_via_fft(x: Sequence, h: Sequence) -> np.ndarray:
    """Convolución lineal utilizando FFT con zero-padding.

    Más rápida para señales largas.
    """
    x = np.asarray(x)
    h = np.asarray(h)
    n = x.size + h.size - 1
    N = int(2 ** np.ceil(np.log2(n)))
    X = np.fft.fft(x, N)
    H = np.fft.fft(h, N)
    y = np.fft.ifft(X * H)[:n]

    if np.isrealobj(x) and np.isrealobj(h):
        y = np.real_if_close(y, tol=1e6)
    return y


def circular_convolution(x: Sequence, h: Sequence) -> np.ndarray:
    """Convolución circular de longitud N = max(len(x), len(h)).

    Si se desea otra longitud, usar FFT con padding manual.
    """
    x = np.asarray(x)
    h = np.asarray(h)
    N = max(x.size, h.size)
    X = np.fft.fft(x, N)
    H = np.fft.fft(h, N)
    y = np.fft.ifft(X * H)
    if np.isrealobj(x) and np.isrealobj(h):
        y = np.real_if_close(y, tol=1e6)
    return y


def cross_correlation(x: Sequence, h: Sequence, conj: bool = True) -> np.ndarray:
    """Correlación cruzada r_xy[n].

    Por convención (cuando las señales pueden ser complejas) se define:
        r_xy[n] = sum_n x[k+n] * conj(y[k])
    Si `conj` es False, se omite el conjugado.

    Implementa la definición discreta "full" retornando longitud nx+ny-1.
    """
    x = np.asarray(x)
    h = np.asarray(h)

    if conj:
        h_inverted = np.conjugate(h[::-1])
    else:
        h_inverted = h[::-1]

    nx = x.size
    nh = h_inverted.size
    n_corr = nx + nh - 1
    result = np.zeros(n_corr, dtype=np.result_type(x, h_inverted))

    for n in range(n_corr):
        s = 0
        for k in range(nx):
            m = n - k
            if 0 <= m < nh:
                s += x[k] * h_inverted[m]
        result[n] = s

    return result


def read_audio_segment(fname):
    sig, sr = sf.read(fname)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    return sig.astype(np.float64), sr


def remove_silence_with_map(sig, sr, frame_len=1024, stride_len=512, thr_db=-30):
    """Quitar silencios de una señal de audio y devolver un mapeo de índices.
    Parametros:
        sig: señal de audio (1D numpy array)
        sr: sample rate (Hz)
        frame_len: longitud de frame para cálculo de energía (muestras)
        stride_len: salto entre frames (muestras)
        thr_db: umbral de energía (dB) para considerar frame como activo
    Devuelve:
      cleaned: señal concatenada de tramos activos
      sample_map: array de la misma longitud que cleaned con el índice original de cada muestra
      segments: lista de (start, end) en muestras originales
      energy: array de energía por frame (dB)
      frame_starts: array de índices de inicio de cada frame
      mask: array booleana de frames activos (True=activo)
    """
    energy = []
    frame_starts = []
    for i in range(0, len(sig) - frame_len + 1, stride_len):
        frame = sig[i:i+frame_len]
        e = 10 * np.log10(np.sum(frame**2) + 1e-12)
        energy.append(e)
        frame_starts.append(i)
    energy = np.array(energy)

    mask = energy > thr_db

    segments = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j+1]:
                j += 1
            start = frame_starts[i]
            end = frame_starts[j] + frame_len

            start = max(0, start)
            end = min(len(sig), end)
            segments.append((start, end))
            i = j + 1
        else:
            i += 1

    cleaned_parts = []
    sample_map_parts = []
    for (s, e) in segments:
        cleaned_parts.append(sig[s:e])
        sample_map_parts.append(np.arange(s, e))

    if len(cleaned_parts) > 0:
        cleaned = np.concatenate(cleaned_parts)
        sample_map = np.concatenate(sample_map_parts)
    else:
        cleaned = np.array([], dtype=sig.dtype)
        sample_map = np.array([], dtype=int)

    return cleaned, sample_map, segments, energy, frame_starts, mask


def normalized_cross_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx = len(x)
    ny = len(y)
    if ny == 0 or nx == 0:
        return np.array([]), np.array([]), np.array([])

    num = signal.correlate(x, y, mode='full')
    lags = np.arange(-(ny-1), nx)

    energy_y = np.sum(y**2)
    if energy_y <= 0:
        denom = np.ones_like(num) * np.finfo(float).eps
        return num / denom, num, lags

    window = np.ones(ny, dtype=np.float64)
    local_energy_x = signal.convolve(x**2, window, mode='full')

    denom = np.sqrt(local_energy_x * energy_y)
    denom_safe = np.where(denom == 0, np.finfo(float).eps, denom)

    ncc = num / denom_safe
    return ncc, num, lags


def read_audio_segment(file_path: str, seconds: float | None = None, start_sec: float = 0.0, mono: bool = False) -> tuple[np.ndarray, int]:
    """Lee un segmento de audio desde un archivo.

    Args:
        file_path: ruta al archivo de audio (wav, flac, etc.).
        seconds: duración en segundos del segmento a leer. Si es None, lee hasta el final.
        start_sec: segundo inicial desde el que empezar a leer (offset en segundos).
        mono: si True convierte el audio a mono (promedio de canales).

    Returns:
        (data, samplerate) donde data es un array numpy tipo float64 y samplerate es int.

    Nota: utiliza pysoundfile (soundfile). La función importa la librería localmente para
    evitar forzar la dependencia al importar este módulo si no se usa audio.
    """
    import soundfile as sf

    # información para calcular frames
    info = sf.info(file_path)
    sr = int(info.samplerate)
    start_frame = int(start_sec * sr)

    if seconds is None:
        data, sr = sf.read(file_path, dtype='float64')
        if start_frame > 0:
            data = data[start_frame:]
    else:
        frames = int(seconds * sr)
        data, sr = sf.read(file_path, start=start_frame,
                           stop=start_frame + frames, dtype='float64')

    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)

    return data, int(sr)


def save_audio(file_path: str, data: np.ndarray, samplerate: int, subtype: str = None) -> None:
    """Guarda un array numpy como archivo de audio.

    Args:
        file_path: ruta de salida (extensión determina el formato, p.ej. .wav, .flac).
        data: array numpy 1D o 2D (canales como segunda dimensión).
        samplerate: tasa de muestreo en Hz.
        subtype: opcional, por ejemplo 'PCM_16' para wav; si es None se elige por defecto.
    """
    import soundfile as sf

    if subtype is None:
        sf.write(file_path, data, samplerate)
    else:
        sf.write(file_path, data, samplerate, subtype=subtype)


if __name__ == "__main__":
    x = [1, 2, 3]
    h = [0, 1, 0.5]
    print("linear:", linear_convolution(x, h))
    print("fft conv:", conv_via_fft(x, h))
    print("circular:", circular_convolution(x, h))
    print("corr:", cross_correlation(x, h))
