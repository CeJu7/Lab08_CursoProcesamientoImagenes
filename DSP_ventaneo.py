import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches
import IPython.display as ipd

# -----------------------------
# Parámetros iniciales
# -----------------------------
FS = 16000
DURATION = 5
FRAME_SIZE = 0.05  # duración de un frame en segundos
FRAME_STRIDE = 0.01 # stride inicial en segundos

# -----------------------------
# Funciones
# -----------------------------
def grabar_voz(fs=FS, duration=DURATION):
    """Graba la voz del usuario y devuelve un array 1D."""
    print("🎤 Grabando tu voz...")
    voz = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    voz = voz.flatten()
    print("✅ Grabación terminada")
    ipd.display(ipd.Audio(voz, rate=fs))
    return voz

def crear_frames(signal, frame_size, stride, fs):
    """Crea frames y frames con ventana Hamming."""
    frame_length = int(frame_size * fs)
    frame_step = int(stride * fs)
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.append(signal, np.zeros(pad_signal_length - signal_length))

    indices = np.tile(np.arange(frame_length), (num_frames,1)) + \
              np.tile(np.arange(0, num_frames*frame_step, frame_step), (frame_length,1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    hamming_window = np.hamming(frame_length)
    frames_windowed = frames * hamming_window
    return frames, frames_windowed, frame_length, frame_step, num_frames

def crear_figura_interactiva(signal, frames, frames_windowed, frame_length, frame_step, num_frames, fs):
    """Crea la figura interactiva con sliders y retorna objetos para update."""
    fig, axs = plt.subplots(3,1, figsize=(15,9))
    plt.subplots_adjust(left=0.1, bottom=0.35, hspace=0.5)

    # Gráfica 1: señal completa con frame destacado
    axs[0].plot(np.arange(len(signal))/fs, signal, color='blue', label='Señal original')
    highlight_rect = patches.Rectangle((0, min(signal)), frame_length/fs, max(signal)-min(signal),
                                       linewidth=0, facecolor='red', alpha=0.3)
    axs[0].add_patch(highlight_rect)
    axs[0].set_title("Señal completa con frame destacado")
    axs[0].set_xlabel("Tiempo [s]")
    axs[0].set_ylabel("Amplitud")
    axs[0].legend(['Señal original','Frame actual'])

    # Gráfica 2: frame seleccionado y Hamming
    line_frame, = axs[1].plot(frames[0], label="Frame sin ventana")
    line_windowed, = axs[1].plot(frames_windowed[0], label="Frame con Hamming", alpha=0.7)
    line_ham, = axs[1].plot(np.hamming(frame_length)*max(frames[0]), label="Ventana Hamming (escalada)",
                            linestyle='--', color='gray')
    axs[1].set_title("Frame seleccionado y efecto de la ventana Hamming")
    axs[1].set_xlabel("Muestras")
    axs[1].set_ylabel("Amplitud")
    axs[1].legend()

    # Gráfica 3: FFT
    fft_vals = np.fft.rfft(frames_windowed[0])
    fft_freqs = np.fft.rfftfreq(frame_length, d=1/fs)
    line_fft, = axs[2].plot(fft_freqs, np.abs(fft_vals))
    axs[2].set_title("FFT del frame con ventana Hamming")
    axs[2].set_xlabel("Frecuencia [Hz]")
    axs[2].set_ylabel("Magnitud")

    # Sliders
    ax_frame = plt.axes([0.1, 0.2, 0.8, 0.03])
    slider_frame = Slider(ax_frame, 'Frame', 0, max(0,num_frames-1), valinit=0, valstep=1)

    ax_stride = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider_stride = Slider(ax_stride, 'Stride [ms]', 1, 50, valinit=FRAME_STRIDE*1000, valstep=1)

    return fig, axs, highlight_rect, line_frame, line_windowed, line_ham, line_fft, slider_frame, slider_stride

def conectar_sliders(slider_frame, slider_stride, signal, frame_size, fs,
                      axs, highlight_rect, line_frame, line_windowed, line_ham, line_fft):
    """Función para actualizar todo al mover los sliders."""
    def update(val):
        stride_sec = slider_stride.val / 1000
        frames, frames_windowed, frame_length, frame_step, num_frames = crear_frames(signal, frame_size, stride_sec, fs)

        i = int(slider_frame.val)
        i = min(i, num_frames-1)
        slider_frame.valmax = max(0, num_frames-1)
        if slider_frame.val > slider_frame.valmax:
            slider_frame.set_val(slider_frame.valmax)

        # Gráfica 1
        highlight_rect.set_x(i*frame_step/fs)
        highlight_rect.set_width(frame_length/fs)

        # Gráfica 2
        line_frame.set_ydata(frames[i])
        line_windowed.set_ydata(frames_windowed[i])
        line_ham.set_ydata(np.hamming(frame_length)*max(frames[i]))
        y_min = min(frames[i].min(), frames_windowed[i].min(), 0)
        y_max = max(frames[i].max(), frames_windowed[i].max(), 0)
        axs[1].set_ylim(y_min*1.1, y_max*1.1)

        # Gráfica 3: FFT
        fft_vals = np.fft.rfft(frames_windowed[i])
        line_fft.set_ydata(np.abs(fft_vals))
        axs[2].set_ylim(0, np.max(np.abs(fft_vals))*1.1)

        fig.canvas.draw_idle()

    slider_frame.on_changed(update)
    slider_stride.on_changed(update)

# -----------------------------
# Programa principal
# -----------------------------
voz = grabar_voz()
frames, frames_windowed, frame_length, frame_step, num_frames = crear_frames(voz, FRAME_SIZE, FRAME_STRIDE, FS)
fig, axs, highlight_rect, line_frame, line_windowed, line_ham, line_fft, slider_frame, slider_stride = crear_figura_interactiva(
    voz, frames, frames_windowed, frame_length, frame_step, num_frames, FS
)
conectar_sliders(slider_frame, slider_stride, voz, FRAME_SIZE, FS,
                 axs, highlight_rect, line_frame, line_windowed, line_ham, line_fft)
plt.show()
