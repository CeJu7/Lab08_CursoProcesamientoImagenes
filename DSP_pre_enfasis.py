import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# -----------------------------
# Parámetros
# -----------------------------
FS = 16000
DURATION = 5
ALPHA_INIT = 0.97

# -----------------------------
# Grabación de voz
# -----------------------------
print("🎤 Grabando tu voz...")
voz = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
sd.wait()
voz = voz.flatten()
print("✅ Grabación terminada")

# -----------------------------
# Función de pre-énfasis
# -----------------------------
def pre_enfasis(signal, alpha):
    return np.append(signal[0], signal[1:] - alpha*signal[:-1])

# -----------------------------
# Crear figura
# -----------------------------
fig, ax = plt.subplots(figsize=(15,5))
plt.subplots_adjust(bottom=0.3)

t = np.arange(len(voz))/FS
line_original, = ax.plot(t, voz, label="Señal original", alpha=0.7)
y_pre_init = pre_enfasis(voz, ALPHA_INIT)
line_pre, = ax.plot(t, y_pre_init, label=f"Pre-énfasis α={ALPHA_INIT}", alpha=0.7)

ax.set_title("Señal original vs pre-énfasis")
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Amplitud")
ax.legend()
ax.grid(True)

# -----------------------------
# Slider para α
# -----------------------------
ax_alpha = plt.axes([0.15, 0.15, 0.7, 0.03])
slider_alpha = Slider(ax_alpha, 'α pre-énfasis', 0.0, 0.99, valinit=ALPHA_INIT, valstep=0.01)

# -----------------------------
# Botón para reproducir pre-énfasis
# -----------------------------
ax_button = plt.axes([0.45, 0.05, 0.1, 0.04])
button_play = Button(ax_button, "Reproducir")

def update(val):
    alpha = slider_alpha.val
    y_pre = pre_enfasis(voz, alpha)
    line_pre.set_ydata(y_pre)
    line_pre.set_label(f"Pre-énfasis α={alpha:.2f}")
    ax.legend()
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

def play_pre(event):
    alpha = slider_alpha.val
    y_pre = pre_enfasis(voz, alpha)
    sd.play(y_pre, FS)
    sd.wait()

slider_alpha.on_changed(update)
button_play.on_clicked(play_pre)

plt.show()
