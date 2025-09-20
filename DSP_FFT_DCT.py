import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.fft import fft, ifft, fftfreq
from scipy.fftpack import dct, idct

# -----------------------------
# 1. Generar señales de ejemplo
# -----------------------------
def generar_senal(tipo='senoidal', N=128, ruido=0.0):
    t = np.linspace(0, 1, N, endpoint=False)
    if tipo == 'senoidal':
        senal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*15*t) + 0.2*np.sin(2*np.pi*30*t)
    elif tipo == 'exponencial':
        senal = np.exp(-3*t) * np.sin(2*np.pi*5*t)
    elif tipo == 'escalon':
        senal = np.zeros(N)
        senal[N//3:2*N//3] = 1.0
    elif tipo == 'impulso':
        senal = np.zeros(N)
        senal[N//2] = 1.0
    elif tipo == 'audio-like':
        senal = np.zeros(N)
        senal[0] = np.random.randn()*0.1
        for i in range(1, N):
            senal[i] = 0.9*senal[i-1] + 0.1*np.random.randn()
    else:
        senal = np.zeros(N)
    senal += ruido*np.random.randn(N)
    return senal

# -----------------------------
# 2. Energía acumulada
# -----------------------------
def energia_acumulada(coef):
    energia_total = np.sum(np.abs(coef)**2)
    energia_cum = np.cumsum(np.abs(coef)**2)
    return energia_cum / energia_total * 100

# -----------------------------
# 3. Reconstrucción FFT equivalente a n_coef_dct
# -----------------------------
def reconstruir_fft(fft_completa, n_coef_dct):
    N = len(fft_completa)
    fft_recon = np.zeros(N, dtype=complex)
    fft_recon[0] = fft_completa[0]
    n_coef_fft = n_coef_dct // 2
    if n_coef_fft > 0:
        fft_recon[1:n_coef_fft+1] = fft_completa[1:n_coef_fft+1]
        fft_recon[-n_coef_fft:] = np.conj(fft_completa[1:n_coef_fft+1][::-1])
    return fft_recon

# -----------------------------
# 4. Función principal con slider
# -----------------------------
def main():
    N = 128
    fs = 128
    tipos = ['senoidal', 'exponencial', 'escalon', 'impulso', 'audio-like']

    for tipo in tipos:
        print(f"\nAnalizando señal: {tipo}")
        senal = generar_senal(tipo, N, ruido=0.05)
        dct_coef = dct(senal, type=2, norm='ortho')
        fft_coef = fft(senal)
        energia_dct = energia_acumulada(dct_coef)
        energia_fft = energia_acumulada(fft_coef)
        t = np.arange(N)

        # -----------------------------
        # Calcular error RMS acumulado (estático)
        # -----------------------------
        rms_dct_acc = [np.sqrt(np.mean((senal - idct(np.concatenate([dct_coef[:k], np.zeros(N-k)]),
                                                         type=2, norm='ortho'))**2)) for k in range(1, N+1)]
        rms_fft_acc = [np.sqrt(np.mean((senal - np.real(ifft(reconstruir_fft(fft_coef, k))))**2)) for k in range(1, N+1)]

        # -----------------------------
        # Crear figura y subplots
        # -----------------------------
        fig = plt.figure(figsize=(14, 10))
        plt.subplots_adjust(bottom=0.25, hspace=0.4)

        ax1 = plt.subplot(3, 2, 1)  # Original
        ax2 = plt.subplot(3, 2, 2)  # Reconstrucción
        ax3 = plt.subplot(3, 2, 3)  # Energía acumulada
        ax4 = plt.subplot(3, 2, 4)  # Error RMS acumulado
        ax5 = plt.subplot(3, 2, 5)  # Espectro FFT
        ax6 = plt.subplot(3, 2, 6)  # Espectro DCT

        # Señal original
        ax1.plot(t, senal, 'b')
        ax1.set_title('Señal Original')
        ax1.grid(True)

        # Reconstrucción inicial
        n_coef_init = 10
        senal_dct = idct(np.concatenate([dct_coef[:n_coef_init], np.zeros(N-n_coef_init)]), type=2, norm='ortho')
        senal_fft = np.real(ifft(reconstruir_fft(fft_coef, n_coef_init)))
        l_orig, = ax2.plot(t, senal, 'k--', label='Original', alpha=0.7)
        l_dct, = ax2.plot(t, senal_dct, 'r', label=f'DCT (RMS={np.sqrt(np.mean((senal-senal_dct)**2)):.3f})')
        l_fft, = ax2.plot(t, senal_fft, 'g', label=f'FFT (RMS={np.sqrt(np.mean((senal-senal_fft)**2)):.3f})')
        ax2.set_title('Reconstrucción DCT vs FFT')
        ax2.legend()
        ax2.grid(True)

        # Energía acumulada
        ax3.plot(energia_dct, 'r', label='DCT')
        ax3.plot(energia_fft, 'g', label='FFT')
        ax3.set_title('Energía acumulada (%)')
        ax3.legend()
        ax3.grid(True)

        # Error RMS acumulado (estático)
        ax4.plot(range(1,N+1), rms_dct_acc, 'r', label='DCT')
        ax4.plot(range(1,N+1), rms_fft_acc, 'g', label='FFT')
        ax4.set_title('Error RMS acumulado vs coeficientes')
        ax4.set_xlabel('Número de coeficientes')
        ax4.set_ylabel('RMS')
        ax4.legend()
        ax4.grid(True)

        # Espectro FFT
        freqs = fftfreq(N, d=1/fs)[:N//2]
        ax5.stem(freqs, np.abs(fft_coef[:N//2]), 'g', basefmt=' ')
        ax5.set_title('Espectro FFT')
        ax5.set_xlabel('Frecuencia [Hz]')
        ax5.grid(True)

        # Espectro DCT
        ax6.stem(np.arange(N), np.abs(dct_coef), 'r', basefmt=' ')
        ax6.set_title('Espectro DCT')
        ax6.set_xlabel('Coeficiente')
        ax6.grid(True)

        # -----------------------------
        # Slider para reconstrucción
        # -----------------------------
        axcoef = plt.axes([0.25, 0.05, 0.50, 0.03])
        slider_coef = Slider(axcoef, 'Coeficientes', 1, N, valinit=n_coef_init, valstep=1)

        def update(val):
            n_coef = int(slider_coef.val)
            senal_dct = idct(np.concatenate([dct_coef[:n_coef], np.zeros(N-n_coef)]), type=2, norm='ortho')
            senal_fft = np.real(ifft(reconstruir_fft(fft_coef, n_coef)))

            rms_dct_val = np.sqrt(np.mean((senal-senal_dct)**2))
            rms_fft_val = np.sqrt(np.mean((senal-senal_fft)**2))

            l_dct.set_ydata(senal_dct)
            l_dct.set_label(f'DCT (RMS={rms_dct_val:.3f})')
            l_fft.set_ydata(senal_fft)
            l_fft.set_label(f'FFT (RMS={rms_fft_val:.3f})')

            ax2.legend()
            fig.canvas.draw_idle()

        slider_coef.on_changed(update)
        plt.suptitle(f'Comparación interactiva DCT vs FFT - {tipo}')
        plt.show()

if __name__ == "__main__":
    main()
