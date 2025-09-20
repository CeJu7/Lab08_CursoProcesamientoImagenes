import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# -----------------------------
# Parámetros principales
# -----------------------------
FS = 16000       # Frecuencia de muestreo [Hz]
DURATION = 5     # Duración de la grabación [s]
NFFT = 1024      # Tamaño de la FFT
NUM_MEL = 10     # Número de filtros Mel
EPSILON = 1e-10  # Para evitar log(0)

# -----------------------------
# Funciones de conversión Mel
# -----------------------------
def hz_to_mel(hz: float) -> float:
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel: float) -> float:
    return 700 * (10**(mel / 2595) - 1)

# -----------------------------
# Banco de filtros Mel
# -----------------------------
def mel_filter_bank(num_filters: int, NFFT: int, fs: int, fmin: float = 0, fmax: float = None):
    if fmax is None:
        fmax = fs / 2

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((NFFT + 1) * hz_points / fs).astype(int)

    filter_bank = np.zeros((num_filters, NFFT // 2 + 1))
    for m in range(1, num_filters + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        filter_bank[m-1, f_m_minus:f_m] = (np.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
        filter_bank[m-1, f_m:f_m_plus] = (f_m_plus - np.arange(f_m, f_m_plus)) / (f_m_plus - f_m)
    return filter_bank, bin_points

# -----------------------------
# Grabación de voz
# -----------------------------
def grabar_voz(fs: int, dur: int) -> np.ndarray:
    print("🎤 Grabando tu voz...")
    voz = sd.rec(int(dur * fs), samplerate=fs, channels=1)
    sd.wait()
    voz = voz.flatten()
    print("✅ Grabación terminada")
    sd.play(voz, fs)
    return voz

# -----------------------------
# Cálculo de espectro y energía Mel
# -----------------------------
def calcular_espectro_y_mel_log(voz: np.ndarray, NFFT: int, num_mel: int, fs: int):
    voz_fft = np.fft.rfft(voz * np.hamming(len(voz)), NFFT)
    voz_power = np.abs(voz_fft)**2
    mel_filters, bin_points = mel_filter_bank(num_mel, NFFT, fs)
    mel_energy = np.dot(mel_filters, voz_power)
    mel_energy_log = np.log(mel_energy + EPSILON)  # evitar log(0)
    return voz_power, mel_filters, mel_energy, mel_energy_log

# -----------------------------
# Visualización del espectro y filtros Mel
# -----------------------------
def mostrar_espectro_y_filtros(voz_power: np.ndarray, mel_filters: np.ndarray, fs: int):
    fig, ax = plt.subplots(figsize=(12, 6))

    freqs = np.linspace(0, fs/2, len(voz_power))
    ax.plot(freqs, voz_power, color='blue', label="Espectro de señal")

    cmap = plt.get_cmap("tab10")
    filter_scale = max(voz_power) * 0.6
    for i in range(mel_filters.shape[0]):
        filter_scaled = mel_filters[i] * filter_scale
        ax.plot(freqs, filter_scaled, linestyle='--', alpha=0.8,
                color=cmap(i % 10), label=f'Filtro Mel {i+1}')

        # Pico del filtro
        peak_idx = np.argmax(filter_scaled)
        ax.plot(freqs[peak_idx], filter_scaled[peak_idx],
                'o', color=cmap(i % 10))
        ax.text(freqs[peak_idx], filter_scaled[peak_idx]*1.05,
                f"{int(freqs[peak_idx])}Hz", ha='center', va='bottom', fontsize=8, color=cmap(i % 10))

    ax.set_title("Espectro de señal y filtros Mel")
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Amplitud")
    ax.legend()
    plt.show()

# -----------------------------
# Visualización de energía Mel (log)
# -----------------------------
def mostrar_energia_mel_log(mel_energy_log: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(np.arange(len(mel_energy_log)), mel_energy_log, color='orange', alpha=0.7)
    ax.set_title("Energía Mel (log)")
    ax.set_xlabel("Filtro Mel")
    ax.set_ylabel("Energía logarítmica")

    # Añadir valores encima de cada barra
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=8)

    plt.show()

# -----------------------------
# Ejecución principal
# -----------------------------
voz = grabar_voz(FS, DURATION)
voz_power, mel_filters, mel_energy, mel_energy_log = calcular_espectro_y_mel_log(
    voz, NFFT, NUM_MEL, FS)

# Mostrar gráficos separados
mostrar_espectro_y_filtros(voz_power, mel_filters, FS)
mostrar_energia_mel_log(mel_energy_log)
