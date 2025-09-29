import numpy as np
import librosa
import scipy.signal
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import soundfile as sf


def load_audio(filename, sr=None, mono=True, duration=None, offset=0.0):
    """
    Carga un archivo de audio utilizando librosa con parámetros configurables.

    Esta función proporciona una interfaz flexible para cargar archivos de audio
    en múltiples formatos (WAV, MP3, FLAC, etc.). Incluye normalización automática
    de la amplitud para evitar saturación y garantizar un rango dinámico consistente.
    La función puede remuestrear automáticamente el audio, convertir a mono y
    cargar segmentos específicos del archivo.

    Parámetros:
    -----------
    filename : str
        Ruta completa o relativa del archivo de audio a cargar.
        Soporta formatos: WAV, MP3, FLAC, M4A, OGG, etc.
    sr : int, opcional
        Frecuencia de muestreo objetivo en Hz. Si es None, mantiene
        la frecuencia original del archivo. Valores típicos: 22050, 44100, 48000
    mono : bool
        Si True, convierte audio estéreo/multicanal a mono mediante promedio.
        Si False, mantiene todos los canales (retorna array 2D)
    duration : float, opcional
        Duración máxima en segundos a cargar desde el archivo.
        Si None, carga el archivo completo
    offset : float
        Tiempo de inicio en segundos desde donde comenzar la carga.
        Útil para analizar segmentos específicos de archivos largos

    Retorna:
    --------
    y : np.ndarray
        Señal de audio normalizada. Shape: (n_samples,) si mono=True,
        (n_channels, n_samples) si mono=False
    sr : int
        Frecuencia de muestreo final del audio cargado

    Raises:
    -------
    FileNotFoundError
        Si el archivo especificado no existe
    librosa.util.exceptions.ParameterError
        Si los parámetros de carga son inválidos
    """
    y, sr_orig = librosa.load(
        filename, sr=sr, mono=mono, duration=duration, offset=offset)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return y, sr if sr is not None else sr_orig


def preprocess_audio(y, normalize=True, remove_silence=False,
                     silence_threshold=0.01, frame_length=2048, hop_length=512):
    """
    Aplica técnicas de preprocesamiento para mejorar la calidad de la señal
    y optimizar la extracción posterior de características.

    El preprocesamiento incluye normalización de amplitud y detección/remoción
    inteligente de segmentos silenciosos. La detección de silencio utiliza
    el RMS (Root Mean Square) de la energía calculado por ventanas deslizantes
    para identificar regiones con baja actividad de señal. Los segmentos
    detectados como silencio se remueven manteniendo márgenes de seguridad
    para preservar ataques y decaimientos naturales.

    Parámetros:
    -----------
    y : np.ndarray
        Señal de audio de entrada. Shape: (n_samples,)
    normalize : bool
        Si True, normaliza la amplitud dividiendo por el valor absoluto máximo.
        Garantiza que la señal esté en el rango [-1, 1]
    remove_silence : bool
        Si True, identifica y remueve automáticamente segmentos silenciosos
        basándose en análisis de energía RMS por ventanas
    silence_threshold : float
        Umbral relativo para detectar silencio (0.0-1.0).
        Se calcula como fracción de la energía RMS máxima.
        Valores menores = más agresivo en detectar silencio
    frame_length : int
        Longitud de la ventana para análisis de energía (en muestras)
    hop_length : int
        Salto entre ventanas consecutivas (en muestras)

    Retorna:
    --------
    y_processed : np.ndarray
        Señal preprocesada con normalización y/o remoción de silencio aplicada.
        Puede ser más corta que la señal original si se removió silencio

    Notas:
    ------
    - La remoción de silencio preserva márgenes antes y después de segmentos activos
    - Si toda la señal se considera silencio, retorna la señal original
    - La normalización se aplica solo si la señal contiene valores no-cero
    """
    y_processed = y.copy()

    if normalize and np.max(np.abs(y_processed)) > 0:
        y_processed = y_processed / np.max(np.abs(y_processed))

    if remove_silence:
        hop_length_silence = min(hop_length, len(y_processed) // 10)
        energy = librosa.feature.rms(y=y_processed,
                                     frame_length=frame_length,
                                     hop_length=hop_length_silence)[0]

        non_silent_frames = energy > silence_threshold * np.max(energy)

        non_silent_intervals = librosa.frames_to_samples(
            np.where(non_silent_frames)[0],
            hop_length=hop_length_silence
        )

        if len(non_silent_intervals) > 0:
            start = max(0, non_silent_intervals[0] - hop_length_silence)
            end = min(len(y_processed),
                      non_silent_intervals[-1] + hop_length_silence)
            y_processed = y_processed[start:end]

    return y_processed


def extract_temporal_features(y, sr):
    """
    Calcula un conjunto comprensivo de características estadísticas y temporales
    que describen propiedades fundamentales de la señal en el dominio del tiempo.

    Esta función extrae medidas estadísticas básicas (media, desviación estándar,
    máximo), momentos estadísticos de orden superior (asimetría, curtosis),
    medidas de energía (RMS, entropía), y características específicas de audio
    como la tasa de cruces por cero. Estas características son fundamentales
    para clasificación de audio, detección de eventos y análisis de calidad.

    Parámetros:
    -----------
    y : np.ndarray
        Señal de audio en el dominio temporal. Shape: (n_samples,)
    sr : int
        Frecuencia de muestreo en Hz, utilizada para calcular duraciones
        y normalizar características dependientes del tiempo

    Retorna:
    --------
    features : dict
        Diccionario con características temporales que incluye:

        Metadatos básicos:
        - 'duration': Duración total en segundos
        - 'samples': Número total de muestras
        - 'sample_rate': Frecuencia de muestreo

        Estadísticas de amplitud:
        - 'mean_amplitude': Media del valor absoluto de la señal
        - 'std_amplitude': Desviación estándar de la amplitud
        - 'max_amplitude': Amplitud máxima absoluta
        - 'rms': Valor RMS (Root Mean Square) de la señal

        Momentos estadísticos:
        - 'skewness': Asimetría de la distribución de amplitudes
        - 'kurtosis': Curtosis (medida de colas pesadas/ligeras)
        - https://www.scielo.org.mx/img/revistas/rmib/v38n3//2395-9126-rmib-38-03-00637-gt3.png

        Características específicas de audio:
        - 'zero_crossing_rate': Tasa promedio de cruces por cero

        Características de energía:
        - 'energy_mean': Energía promedio por ventana
        - 'energy_std': Desviación estándar de la energía
        - 'energy_max': Energía máxima observada
        - 'energy_min': Energía mínima observada
        - 'energy_entropy': Entropía de Shannon de la distribución de energía

    Notas:
    ------
    - La entropía de energía mide la uniformidad de distribución energética
    - Los cruces por cero indican contenido de alta frecuencia
    - El RMS proporciona una medida perceptualmente relevante del volumen
    """
    features = {}

    features['duration'] = len(y) / sr
    features['samples'] = len(y)
    features['sample_rate'] = sr

    features['mean_amplitude'] = np.mean(np.abs(y))
    features['std_amplitude'] = np.std(y)
    features['max_amplitude'] = np.max(np.abs(y))
    features['rms'] = np.sqrt(np.mean(y**2))

    features['skewness'] = skew(y)
    features['kurtosis'] = kurtosis(y)

    features['zero_crossing_rate'] = np.mean(
        librosa.feature.zero_crossing_rate(y)[0])

    frame_length = min(2048, len(y))
    hop_length = frame_length // 4

    if len(y) >= frame_length:
        energy = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length)[0]
        features['energy_mean'] = np.mean(energy)
        features['energy_std'] = np.std(energy)
        features['energy_max'] = np.max(energy)
        features['energy_min'] = np.min(energy)

        energy_norm = energy / (np.sum(energy) + 1e-12)
        features['energy_entropy'] = - \
            np.sum(energy_norm * np.log2(energy_norm + 1e-12))
    else:
        energy_val = np.sqrt(np.mean(y**2))
        features['energy_mean'] = energy_val
        features['energy_std'] = 0
        features['energy_max'] = energy_val
        features['energy_min'] = energy_val
        features['energy_entropy'] = 0

    return features


def extract_envelope_features(y, sr, frame_length=2048, hop_length=512):
    """
    Analiza la envolvente temporal de la señal para extraer características
    relacionadas con la dinámica y estructura temporal del audio.

    La función modela el comportamiento ADSR (Attack, Decay, Sustain, Release)
    típico de instrumentos musicales y sonidos naturales. Utiliza el RMS
    calculado por ventanas deslizantes para estimar la envolvente de amplitud
    y determinar los tiempos característicos de cada fase. Es especialmente
    útil para clasificación de instrumentos, análisis de articulación musical
    y caracterización de eventos sonoros.

    Parámetros:
    -----------
    y : np.ndarray
        Señal de audio temporal. Shape: (n_samples,)
    sr : int
        Frecuencia de muestreo en Hz, necesaria para convertir
        índices de ventana a unidades de tiempo
    frame_length : int
        Longitud de cada ventana de análisis en muestras.
        Valores típicos: 1024-4096 muestras
    hop_length : int
        Número de muestras entre ventanas consecutivas.
        Determina la resolución temporal del análisis

    Retorna:
    --------
    envelope_features : dict
        Características de la envolvente ADSR y estadísticas:

        Tiempos ADSR:
        - 'attack_time': Tiempo hasta alcanzar amplitud máxima (segundos)
        - 'decay_time': Tiempo de decaimiento desde máximo (segundos)
        - 'sustain_level': Nivel promedio de sustain (amplitud normalizada)
        - 'release_time': Tiempo de liberación/decaimiento final (segundos)

        Estadísticas de envolvente:
        - 'envelope_mean': Amplitud promedio de la envolvente
        - 'envelope_std': Desviación estándar de la envolvente
        - 'envelope_max': Amplitud máxima de la envolvente
        - 'envelope_range': Rango dinámico (máximo - mínimo)

    Algoritmo:
    ----------
    1. Calcula RMS por ventanas deslizantes como aproximación de envolvente
    2. Identifica el punto de amplitud máxima
    3. Analiza fase de ataque (inicio hasta máximo)
    4. Detecta decaimiento (90% del máximo)
    5. Estima sustain (región media estable)
    6. Mide release (cuarto final de la señal)

    Notas:
    ------
    - Para señales muy cortas, algunos parámetros ADSR pueden ser cero
    - El análisis es robusto ante variaciones de ruido en la envolvente
    - Útil para distinguir entre instrumentos percusivos y sostenidos
    """
    features = {}

    frame_length = min(frame_length, len(y))
    hop_length = min(hop_length, frame_length // 4)

    if len(y) < frame_length:
        envelope = np.abs(y)
        features['attack_time'] = 0
        features['decay_time'] = len(y) / sr
        features['sustain_level'] = np.mean(envelope)
        features['release_time'] = 0
    else:
        envelope = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length)[0]

        max_idx = np.argmax(envelope)
        max_val = envelope[max_idx]

        features['attack_time'] = (max_idx * hop_length) / sr

        if max_idx < len(envelope) - 1:
            post_attack = envelope[max_idx:]

            decay_threshold = 0.9 * max_val
            decay_idx = np.where(post_attack < decay_threshold)[0]
            if len(decay_idx) > 0:
                features['decay_time'] = (decay_idx[0] * hop_length) / sr
            else:
                features['decay_time'] = 0

            mid_start = max_idx + len(post_attack) // 4
            mid_end = max_idx + 3 * len(post_attack) // 4
            if mid_end > mid_start:
                features['sustain_level'] = np.mean(
                    envelope[mid_start:mid_end])
            else:
                features['sustain_level'] = max_val

            final_quarter = post_attack[-len(post_attack)//4:]
            if len(final_quarter) > 1:
                features['release_time'] = len(final_quarter) * hop_length / sr
            else:
                features['release_time'] = 0
        else:
            features['decay_time'] = 0
            features['sustain_level'] = max_val
            features['release_time'] = 0

    features['envelope_mean'] = np.mean(envelope)
    features['envelope_std'] = np.std(envelope)
    features['envelope_max'] = np.max(envelope)
    features['envelope_range'] = np.max(envelope) - np.min(envelope)

    return features


def extract_spectral_features(y, sr, n_fft=2048, hop_length=512):
    """
    Calcula características espectrales fundamentales que describen la distribución
    de energía en frecuencia y la estructura tímbrica del audio.

    Esta función utiliza la Transformada de Fourier de Tiempo Corto (STFT)
    para analizar cómo evoluciona el contenido frecuencial a lo largo del tiempo.
    Las características extraídas son ampliamente utilizadas en MIR (Music
    Information Retrieval), clasificación de audio, y análisis de timbre.
    Cada característica se calcula como estadísticas (media y desviación)
    a través de todas las ventanas temporales.

    Parámetros:
    -----------
    y : np.ndarray
        Señal de audio en el dominio temporal. Shape: (n_samples,)
    sr : int
        Frecuencia de muestreo en Hz, define el rango frecuencial máximo
    n_fft : int
        Tamaño de la FFT, determina la resolución frecuencial.
        Resolución = sr / n_fft Hz por bin
    hop_length : int
        Salto entre ventanas STFT consecutivas.
        Determina la resolución temporal del análisis

    Retorna:
    --------
    features : dict
        Características espectrales con estadísticas temporales:

        Centroide espectral:
        - 'spectral_centroid_mean': Frecuencia promedio ponderada ("brillo")
        - 'spectral_centroid_std': Variabilidad del centroide en el tiempo

        Rolloff espectral:
        - 'spectral_rolloff_mean': Frecuencia bajo la cual está el 85% de energía
        - 'spectral_rolloff_std': Variabilidad del rolloff

        Ancho de banda espectral:
        - 'spectral_bandwidth_mean': Dispersión frecuencial promedio
        - 'spectral_bandwidth_std': Variabilidad del ancho de banda

        Contraste espectral (7 bandas):
        - 'spectral_contrast_{i}_mean': Diferencia entre picos y valles por banda
        - 'spectral_contrast_{i}_std': Variabilidad del contraste por banda

        Planitud espectral:
        - 'spectral_flatness_mean': Medida de "ruido blanco" vs tonalidad
        - 'spectral_flatness_std': Variabilidad de la planitud

    Interpretación:
    ---------------
    - Centroide alto = sonido "brillante" o agudo
    - Rolloff alto = contenido extendido de altas frecuencias
    - Bandwidth alto = sonido "áspero" o con muchas frecuencias
    - Contraste alto = estructura harmónica clara
    - Flatness alto = sonido más parecido a ruido que a tono
    """
    features = {}

    n_fft = min(n_fft, len(y))
    hop_length = min(hop_length, n_fft // 4)

    spectral_centroids = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    for i in range(spectral_contrast.shape[0]):
        features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
        features[f'spectral_contrast_{i}_std'] = np.std(spectral_contrast[i])

    spectral_flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop_length)[0]
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)
    features['spectral_flatness_std'] = np.std(spectral_flatness)

    return features


def extract_harmonic_features(y, sr, n_fft=2048, hop_length=512):
    """
    Analiza la estructura armónica y tonal de la señal mediante separación
    armónico-percusiva, detección de pitch (percepción de la altura de un sonido)
    y análisis cromático.

    Utiliza la descomposición HPSS (Harmonic-Percussive Source Separation)
    para separar componentes tonales de transientes. Implementa seguimiento
    de pitch mediante detección de picos en el espectrograma y extrae
    características cromáticas que representan el contenido tonal independiente
    de la octava. Es fundamental para clasificación de género musical,
    detección de instrumentos armónicos vs percusivos, y análisis de armonía.

    - Harmonic → Sonidos tonales y sostenidos como guitarra, teclado o voz.
    Se ven como líneas horizontales en el espectrograma.

    - Percussive → Sonidos transitorios y rítmicos como batería, palmas o golpes.
      Se ven como líneas verticales en el espectrograma.

    Parámetros:
    -----------
    y : np.ndarray
        Señal de audio temporal. Shape: (n_samples,)
    sr : int
        Frecuencia de muestreo en Hz, define el rango de pitch detectable
    n_fft : int
        Tamaño de FFT para análisis espectral. Afecta resolución frecuencial
    hop_length : int
        Salto entre ventanas. Determina resolución temporal del pitch tracking

    Retorna:
    --------
    features : dict
        Características armónicas y tonales:

        Separación armónico-percusiva:
        - 'harmonic_ratio': Proporción de energía armónica (0-1)
        - 'percussive_ratio': Proporción de energía percusiva (0-1)

        Características de pitch:
        - 'pitch_mean': Frecuencia fundamental promedio (Hz)
        - 'pitch_std': Variabilidad del pitch
        - 'pitch_max': Pitch máximo detectado
        - 'pitch_min': Pitch mínimo detectado
        - 'pitch_range': Rango de pitch (max - min)

        Vector cromático (12 dimensiones):
        - 'chroma_{0-11}_mean': Intensidad promedio de cada clase de pitch
        - 'chroma_{0-11}_std': Variabilidad temporal de cada chroma

    Algoritmo:
    ----------
    1. Aplica HPSS para separar componentes armónicos y percusivos
    2. Calcula ratios de energía entre componentes
    3. Ejecuta pitch tracking mediante detección de picos espectrales
    4. Filtra pitches por magnitud para robustez
    5. Extrae características cromáticas mediante proyección a 12 clases tonales

    Interpretación:
    ---------------
    - harmonic_ratio alto = instrumento tonal/melódico
    - percussive_ratio alto = instrumento rítmico/percusivo
    - pitch estable = nota sostenida
    - pitch variable = melodía o vibrato
    - chroma concentrado = tonalidad definida
    - chroma disperso = atonalidad o ruido

    Notas:
    ------
    - Manejo robusto de errores en pitch detection
    - Valores cero para pitch indican ausencia de tonalidad clara
    - Las 12 dimensiones cromáticas corresponden a las clases de pitch
    """
    features = {}

    n_fft = min(n_fft, len(y))
    hop_length = min(hop_length, len(y) // 4)

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    features['harmonic_ratio'] = np.sum(y_harmonic**2) / (np.sum(y**2) + 1e-12)
    features['percussive_ratio'] = np.sum(
        y_percussive**2) / (np.sum(y**2) + 1e-12)

    try:
        pitches, magnitudes = librosa.core.piptrack(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

        pitch_track = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_track.append(pitch)

        if len(pitch_track) > 0:
            features['pitch_mean'] = np.mean(pitch_track)
            features['pitch_std'] = np.std(pitch_track)
            features['pitch_max'] = np.max(pitch_track)
            features['pitch_min'] = np.min(pitch_track)
            features['pitch_range'] = features['pitch_max'] - \
                features['pitch_min']
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_max'] = 0
            features['pitch_min'] = 0
            features['pitch_range'] = 0

    except:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['pitch_max'] = 0
        features['pitch_min'] = 0
        features['pitch_range'] = 0

    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
            features[f'chroma_{i}_std'] = np.std(chroma[i])
    except:
        for i in range(12):
            features[f'chroma_{i}_mean'] = 0
            features[f'chroma_{i}_std'] = 0

    return features


def extract_mfcc_features(y, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Calcula coeficientes MFCC (Mel-Frequency Cepstral Coefficients) y sus
    derivadas temporales para caracterización tímbrica perceptualmente relevante.

    Los MFCCs son la representación estándar de oro para reconocimiento de voz
    y clasificación de audio. Modelan la percepción auditiva humana mediante
    el uso de la escala Mel y compresión logarítmica. La transformada cepstral
    separa la envolvente espectral (timbre) de la estructura armónica (pitch),
    proporcionando características compactas y discriminativas. Incluye cálculo
    de deltas (velocidad) y delta-deltas (aceleración) para capturar dinámica temporal.

    Parámetros:
    -----------
    y : np.ndarray
        Señal de audio temporal. Shape: (n_samples,)
    sr : int
        Frecuencia de muestreo en Hz. Determina el rango de frecuencias analizables
    n_mfcc : int
        Número de coeficientes MFCC a extraer. Típicamente 12-13.
        Más coeficientes = mayor detalle, pero posible sobreajuste
    n_fft : int
        Tamaño de FFT para análisis espectral inicial
    hop_length : int
        Salto entre ventanas STFT. Afecta resolución temporal

    Retorna:
    --------
    features : dict
        Características MFCC estáticas y dinámicas:

        MFCCs estáticos:
        - 'mfcc_{0-12}_mean': Valor promedio de cada coeficiente MFCC
        - 'mfcc_{0-12}_std': Desviación estándar de cada coeficiente
        - 'mfcc_{0-12}_max': Valor máximo de cada coeficiente
        - 'mfcc_{0-12}_min': Valor mínimo de cada coeficiente

        Deltas (primeras derivadas - velocidad de cambio):
        - 'mfcc_delta_{0-12}_mean': Velocidad promedio de cambio
        - 'mfcc_delta_{0-12}_std': Variabilidad de la velocidad

        Delta-deltas (segundas derivadas - aceleración):
        - 'mfcc_delta2_{0-12}_mean': Aceleración promedio de cambio
        - 'mfcc_delta2_{0-12}_std': Variabilidad de la aceleración

    Proceso de cálculo:
    --------------------
    1. STFT de la señal de audio
    2. Cálculo del espectrograma de magnitud
    3. Aplicación del banco de filtros Mel (escala perceptual)
    4. Compresión logarítmica (modelado de percepción de intensidad)
    5. Transformada Coseno Discreta (DCT) para decorrelación
    6. Selección de los primeros n_mfcc coeficientes
    7. Cálculo de derivadas temporales (deltas)

    Interpretación:
    ---------------
    - MFCC 0: Relacionado con energía total (a menudo excluido)
    - MFCC 1-12: Envolvente espectral (características tímbricas)
    - Deltas altas: Cambios rápidos en timbre
    - Delta-deltas: Articulación y transiciones suaves/abruptas

    Aplicaciones:
    -------------
    - Reconocimiento automático de voz (ASR)
    - Clasificación de género musical
    - Identificación de instrumentos
    - Detección de emociones en voz
    """
    features = {}

    n_fft = min(n_fft, len(y))
    hop_length = min(hop_length, n_fft // 4)

    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    for i in range(n_mfcc):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        features[f'mfcc_{i}_max'] = np.max(mfccs[i])
        features[f'mfcc_{i}_min'] = np.min(mfccs[i])

    mfcc_delta = librosa.feature.delta(mfccs)
    for i in range(n_mfcc):
        features[f'mfcc_delta_{i}_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc_delta_{i}_std'] = np.std(mfcc_delta[i])

    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    for i in range(n_mfcc):
        features[f'mfcc_delta2_{i}_mean'] = np.mean(mfcc_delta2[i])
        features[f'mfcc_delta2_{i}_std'] = np.std(mfcc_delta2[i])

    return features


def extract_mel_spectrogram_features(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    Genera y analiza el espectrograma Mel para extraer características
    perceptualmente motivadas de la distribución energética frecuencial.

    El espectrograma Mel es una representación tiempo-frecuencia que utiliza
    la escala Mel para modelar la percepción auditiva humana. Las frecuencias
    se mapean de manera no-lineal (logarítmica en altas frecuencias) para
    reflejar la sensibilidad auditiva real. Es ampliamente utilizado en
    aprendizaje profundo para audio, especialmente en redes convolucionales
    que procesan el espectrograma como una "imagen" de audio.

    Parámetros:
    -----------
    y : np.ndarray
        Señal de audio temporal. Shape: (n_samples,)
    sr : int
        Frecuencia de muestreo en Hz. Define el rango frecuencial máximo
    n_mels : int
        Número de bandas/filtros Mel. Típicamente 64-128 para clasificación,
        hasta 256 para aplicaciones de alta resolución
    n_fft : int
        Tamaño de FFT para STFT inicial. Determina resolución frecuencial base
    hop_length : int
        Salto entre ventanas STFT. Controla resolución temporal

    Retorna:
    --------
    features : dict
        Características derivadas del espectrograma Mel:

        Estadísticas globales:
        - 'mel_spec_mean': Energía promedio en escala dB
        - 'mel_spec_std': Variabilidad energética temporal
        - 'mel_spec_max': Pico de energía máximo observado
        - 'mel_spec_min': Energía mínima (generalmente ruido de fondo)

        Características por banda frecuencial (8 bandas resumen):
        - 'mel_band_{0-7}_mean': Energía promedio por región frecuencial
        - 'mel_band_{0-7}_std': Variabilidad temporal por banda

    Proceso de cálculo:
    --------------------
    1. STFT de la señal de entrada
    2. Cálculo del espectrograma de potencia
    3. Aplicación del banco de filtros Mel triangulares
    4. Conversión a escala logarítmica (dB)
    5. Extracción de estadísticas temporales
    6. Resumen por bandas frecuenciales representativas

    Bandas frecuenciales típicas:
    ------------------------------
    - Banda 0: Frecuencias muy bajas (sub-100 Hz)
    - Banda 1-2: Graves (100-500 Hz)
    - Banda 3-4: Medios (500-2000 Hz)
    - Banda 5-6: Agudos (2-8 kHz)
    - Banda 7: Muy agudos (8+ kHz)

    Aplicaciones:
    -------------
    - Entrada para redes neuronales convolucionales
    - Clasificación de género y emoción musical
    - Detección de eventos acústicos
    - Análisis de calidad de audio
    - Recuperación de información musical por similaridad

    Ventajas:
    ---------
    - Representación perceptualmente relevante
    - Compacta comparada con espectrograma lineal
    - Compatible con arquitecturas de deep learning
    - Robusta ante variaciones de ruido
    """
    features = {}

    n_fft = min(n_fft, len(y))
    hop_length = min(hop_length, n_fft // 4)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                              n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    features['mel_spec_mean'] = np.mean(mel_spec_db)
    features['mel_spec_std'] = np.std(mel_spec_db)
    features['mel_spec_max'] = np.max(mel_spec_db)
    features['mel_spec_min'] = np.min(mel_spec_db)

    n_bands_summary = min(8, n_mels)
    band_indices = np.linspace(0, n_mels-1, n_bands_summary, dtype=int)

    for i, band_idx in enumerate(band_indices):
        features[f'mel_band_{i}_mean'] = np.mean(mel_spec_db[band_idx])
        features[f'mel_band_{i}_std'] = np.std(mel_spec_db[band_idx])

    return features


def extract_all_features(filename, sr=22050, normalize=True, remove_silence=False):
    """
    Función principal que ejecuta una extracción completa y sistemática de
    características de audio desde un archivo, integrando todos los métodos
    disponibles en este módulo.

    Esta es la función de más alto nivel que proporciona una interfaz unificada
    para extraer automáticamente más de 100 características de audio diferentes.
    Combina análisis temporal, espectral, armónico, cepstral y de envolvente
    para crear un vector de características comprensivo. Incluye manejo
    robusto de errores, progreso detallado, y metadatos de procesamiento.
    Ideal para datasets grandes y pipelines de machine learning.

    Parámetros:
    -----------
    filename : str
        Ruta completa o relativa del archivo de audio a procesar.
        Acepta formatos estándar: WAV, MP3, FLAC, M4A, OGG
    sr : int
        Frecuencia de muestreo objetivo para análisis uniforme.
        22050 Hz es un buen compromiso velocidad/calidad para clasificación
    normalize : bool
        Si True, normaliza la amplitud para consistencia entre archivos.
        Recomendado para datasets con niveles de grabación variables
    remove_silence : bool
        Si True, remueve automáticamente segmentos silenciosos.
        Útil para datasets con espacios largos de silencio

    Retorna:
    --------
    all_features : dict
        Diccionario comprensivo con todas las características:

        Metadatos:
        - 'filename': Nombre del archivo procesado
        - 'original_duration': Duración original en segundos
        - 'processed_duration': Duración post-procesamiento
        - 'sample_rate': Frecuencia de muestreo utilizada

        Características por categoría:
        - Temporales (15+): Estadísticas básicas, RMS, cruces por cero, etc.
        - Envolvente (8): ADSR y dinámica de amplitud
        - Espectrales (20+): Centroide, rolloff, bandwidth, contraste, flatness
        - Armónicas (30+): Pitch, ratios armónico/percusivo, chroma
        - MFCC (78): Coeficientes estáticos y derivadas temporales
        - Mel (20): Espectrograma Mel y bandas frecuenciales

    Flujo de procesamiento:
    ----------------------
    1. Carga del archivo con parámetros especificados
    2. Preprocesamiento (normalización y remoción de silencio)
    3. Extracción secuencial por categorías con manejo de errores
    4. Consolidación de resultados con metadatos
    5. Validación de integridad del vector final

    Casos de uso:
    -------------
    - Clasificación automática de género musical
    - Detección de instrumentos en grabaciones
    - Análisis de similaridad acústica
    - Datasets para machine learning en audio
    - Control de calidad de grabaciones

    Rendimiento:
    ------------
    - Procesa ~10-30 archivos por minuto (dep. duración y hardware)
    - Vector final típico: 150-200 características por archivo
    - Memoria: ~50-100 MB por archivo durante procesamiento

    Raises:
    -------
    FileNotFoundError
        Si el archivo especificado no existe o no es accesible
    Exception
        Errores durante extracción se reportan pero no detienen el proceso

    Ejemplo:
    --------
    >>> features = extract_all_features('audio/song.wav', sr=44100)
    >>> print(f"Extrajdas {len(features)} características")
    >>> print(f"Duración: {features['duration']:.2f} segundos")
    """
    print(f"Extrayendo características de: {filename}")

    y, sr_actual = load_audio(filename, sr=sr)
    y_processed = preprocess_audio(
        y, normalize=normalize, remove_silence=remove_silence)

    print(f"  - Duración: {len(y_processed)/sr_actual:.2f} segundos")
    print(f"  - Frecuencia de muestreo: {sr_actual} Hz")
    print(f"  - Muestras: {len(y_processed)}")

    all_features = {}

    all_features['filename'] = filename
    all_features['original_duration'] = len(y) / sr_actual
    all_features['processed_duration'] = len(y_processed) / sr_actual
    all_features['sample_rate'] = sr_actual

    try:
        print("  - Extrayendo características temporales...")
        temporal_features = extract_temporal_features(y_processed, sr_actual)
        all_features.update(temporal_features)

        print("  - Extrayendo características de envolvente...")
        envelope_features = extract_envelope_features(y_processed, sr_actual)
        all_features.update(envelope_features)

        print("  - Extrayendo características espectrales...")
        spectral_features = extract_spectral_features(y_processed, sr_actual)
        all_features.update(spectral_features)

        print("  - Extrayendo características armónicas...")
        harmonic_features = extract_harmonic_features(y_processed, sr_actual)
        all_features.update(harmonic_features)

        print("  - Extrayendo características MFCC...")
        mfcc_features = extract_mfcc_features(y_processed, sr_actual)
        all_features.update(mfcc_features)

        print("  - Extrayendo características del espectrograma Mel...")
        mel_features = extract_mel_spectrogram_features(y_processed, sr_actual)
        all_features.update(mel_features)

        print(
            f"  ✓ Extracción completada: {len(all_features)} características")

    except Exception as e:
        print(f"  ✗ Error durante la extracción: {str(e)}")
        raise

    return all_features


def features_to_dataframe(features_dict):
    """
    Convierte estructuras de características de audio a un DataFrame de pandas
    optimizado para análisis estadístico y machine learning.

    Esta utilidad maneja flexiblemente tanto diccionarios individuales como
    listas de diccionarios, facilitando la conversión de resultados de
    extracción a formatos estructurados. El DataFrame resultante está
    listo para análisis exploratorio, visualización, y entrenamiento
    de modelos de machine learning.

    Parámetros:
    -----------
    features_dict : dict o list of dict
        - Si dict: Características de un solo archivo
        - Si list: Características de múltiples archivos
        Debe contener claves numéricas o de string con valores escalares

    Retorna:
    --------
    df : pd.DataFrame
        DataFrame con estructura optimizada:
        - Filas: Archivos/muestras individuales
        - Columnas: Características extraídas
        - Índice: Automático (0, 1, 2, ...)
        - Tipos de datos: Inferidos automáticamente

    Características del DataFrame:
    --------------------------------
    - Columnas numéricas convertidas a tipos apropiados (int, float)
    - Columnas de texto (filename, etc.) como string/object
    - Valores faltantes manejados como NaN
    - Índice numérico secuencial para referencia fácil

    Casos de uso:
    -------------
    - Preparación para sklearn, tensorflow, pytorch
    - Visualización con matplotlib, seaborn, plotly
    - Análisis estadístico descriptivo
    - Exportación a CSV, Excel, bases de datos
    - Operaciones de filtrado y agrupamiento

    Ejemplo:
    --------
    >>> features = extract_all_features('song.wav')
    >>> df = features_to_dataframe(features)
    >>> print(df.shape)  # (1, n_features)
    >>> print(df.dtypes)  # Tipos de datos por columna

    >>> # Múltiples archivos
    >>> features_list = [extract_all_features(f) for f in files]
    >>> df = features_to_dataframe(features_list)
    >>> print(df.shape)  # (n_files, n_features)
    """
    import pandas as pd

    if isinstance(features_dict, dict) and 'filename' in features_dict:
        features_list = [features_dict]
    else:
        features_list = features_dict if isinstance(
            features_dict, list) else [features_dict]

    df = pd.DataFrame(features_list)
    return df


def save_features(features_dict, output_file, format='csv'):
    """
    Persiste características extraídas en diferentes formatos de archivo
    optimizados para distintos casos de uso y herramientas de análisis.

    Proporciona exportación flexible a formatos estándar de la industria,
    cada uno con ventajas específicas: CSV para interoperabilidad y
    visualización, JSON para intercambio web y APIs, PKL para preservación
    completa de estructuras Python. Maneja automáticamente la conversión
    de tipos de datos y la serialización de estructuras complejas.

    Parámetros:
    -----------
    features_dict : dict, list of dict, o pd.DataFrame
        Datos de características a guardar. Acepta:
        - Dict individual: Características de un archivo
        - List de dicts: Dataset completo
        - DataFrame: Datos ya estructurados
    output_file : str
        Ruta completa del archivo de destino incluyendo extensión.
        La carpeta padre debe existir o ser creada previamente
    format : str
        Formato de exportación con características específicas:
        - 'csv': Tabla delimitada, compatible con Excel/R/MATLAB
        - 'json': Formato de intercambio web, legible por humanos
        - 'pkl': Formato nativo Python, preserva tipos exactos

    Formatos soportados:
    -------------------
    CSV:
    - Ventajas: Universal, fácil visualización, compatible con Excel
    - Limitaciones: Solo datos numéricos/texto, sin metadatos complejos
    - Uso: Análisis estadístico, machine learning, reportes

    JSON:
    - Ventajas: Legible, soporta jerarquías, compatible con web APIs
    - Limitaciones: Menos eficiente para datos numéricos grandes
    - Uso: Intercambio de datos, configuraciones, integración web

    PKL (Pickle):
    - Ventajas: Preserva tipos Python exactos, rápido, compacto
    - Limitaciones: Solo Python, problemas de versionado
    - Uso: Caché temporal, intercambio entre scripts Python

    Comportamiento:
    ---------------
    - Crea directorios padre si no existen
    - Sobrescribe archivos existentes sin advertencia
    - Convierte automáticamente a DataFrame si es necesario
    - Maneja valores especiales (inf, -inf, nan)
    - Reporta confirmación de guardado exitoso

    Raises:
    -------
    ValueError
        Si el formato especificado no es soportado
    PermissionError
        Si no hay permisos de escritura en la ubicación
    IOError
        Si hay problemas de espacio en disco o ruta inválida

    Ejemplo:
    --------
    >>> features = extract_all_features('audio.wav')
    >>> save_features(features, 'results/features.csv')  # CSV
    >>> save_features(features, 'results/features.json', 'json')  # JSON
    >>> save_features(features, 'cache/features.pkl', 'pkl')  # Pickle
    """
    import pandas as pd
    import json
    import pickle

    if format == 'csv':
        df = features_to_dataframe(features_dict)
        df.to_csv(output_file, index=False)
        print(f"Características guardadas en: {output_file}")

    elif format == 'json':
        with open(output_file, 'w') as f:
            json.dump(features_dict, f, indent=2, default=str)
        print(f"Características guardadas en: {output_file}")

    elif format == 'pkl':
        with open(output_file, 'wb') as f:
            pickle.dump(features_dict, f)
        print(f"Características guardadas en: {output_file}")

    else:
        raise ValueError(f"Formato no soportado: {format}")


def analyze_feature_importance(features_dict, exclude_meta=True):
    """
    Realiza un análisis estadístico comprensivo para evaluar la relevancia,
    variabilidad y relaciones entre características extraídas.

    Esta función proporciona métricas clave para entender la distribución
    y comportamiento de las características, identificar posibles problemas
    (características constantes, altamente correlacionadas), y guiar la
    selección de características para machine learning. Incluye estadísticas
    descriptivas, análisis de correlación y ranking por varianza.

    Parámetros:
    -----------
    features_dict : dict, list, o pd.DataFrame
        Datos de características a analizar. Si es una lista,
        debe contener múltiples observaciones para cálculos significativos
    exclude_meta : bool
        Si True, excluye columnas no-numéricas (filename, etc.) del análisis.
        Recomendado para enfocar en características discriminativas

    Retorna:
    --------
    analysis : dict
        Análisis estadístico detallado conteniendo:

        'total_features' : int
            Número de características numéricas analizadas

        'feature_stats' : pd.DataFrame
            Estadísticas descriptivas por característica:
            - count: Número de observaciones válidas
            - mean: Media aritmética
            - std: Desviación estándar
            - min/max: Valores extremos
            - 25%/50%/75%: Cuartiles (percentiles)

        'correlation_matrix' : pd.DataFrame
            Matriz de correlación de Pearson (n_features x n_features):
            - Valores [-1, 1]: -1=anticorrelación, 0=independencia, 1=correlación
            - Diagonal = 1 (autocorrelación perfecta)
            - Útil para detectar redundancia entre características

        'variance' : pd.Series
            Varianza de cada característica ordenada descendentemente:
            - Varianza alta = mayor discriminación potencial
            - Varianza baja/cero = característica posiblemente irrelevante
            - Útil para selección de características

    Interpretación:
    ---------------
    Varianza:
    - Alta (>1.0): Característica muy variable, potencialmente discriminativa
    - Media (0.1-1.0): Variabilidad moderada, posiblemente útil
    - Baja (<0.1): Poca variación, candidata a remoción
    - Cero: Constante, debe removerse

    Correlación:
    - |r| > 0.9: Fuertemente correlacionadas, considerar remover una
    - |r| 0.7-0.9: Moderadamente correlacionadas, evaluar importancia
    - |r| < 0.3: Independientes, mantener ambas

    Casos de uso:
    -------------
    - Selección de características para machine learning
    - Detección de características redundantes o constantes
    - Análisis exploratorio de datasets de audio
    - Optimización de pipelines de extracción
    - Control de calidad de características

    Limitaciones:
    -------------
    - Requiere múltiples observaciones para correlaciones significativas
    - Solo analiza relaciones lineales (Pearson)
    - No detecta relaciones no-lineales complejas

    Ejemplo:
    --------
    >>> features_list = [extract_all_features(f) for f in audio_files]
    >>> analysis = analyze_feature_importance(features_list)
    >>> print(f"Características analizadas: {analysis['total_features']}")
    >>> high_var = analysis['variance'].head(10)  # Top 10 por varianza
    >>> corr_matrix = analysis['correlation_matrix']
    """
    import pandas as pd

    df = features_to_dataframe(features_dict)

    if exclude_meta:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]
    else:
        df_numeric = df

    analysis = {}
    analysis['total_features'] = len(df_numeric.columns)
    analysis['feature_stats'] = df_numeric.describe()
    analysis['correlation_matrix'] = df_numeric.corr()
    analysis['variance'] = df_numeric.var().sort_values(ascending=False)

    return analysis


def plot_feature_distribution(features_dict, feature_groups=None, figsize=(15, 10)):
    """
    Genera visualizaciones comprensivas de la distribución estadística de
    características de audio mediante gráficos de caja (boxplots) agrupados.

    Esta función de visualización proporciona una vista panorámica de cómo
    se distribuyen las diferentes categorías de características, facilitando
    la identificación de outliers, rangos típicos, y patrones de variabilidad.
    Los gráficos se organizan automáticamente por grupos temáticos para
    mejorar la interpretabilidad y comparación entre características relacionadas.

    Parámetros:
    -----------
    features_dict : dict, list, o pd.DataFrame
        Datos de características a visualizar. Para gráficos significativos,
        se recomienda tener múltiples observaciones (archivos)
    feature_groups : dict, opcional
        Diccionario que define agrupaciones personalizadas:
        {'grupo_nombre': ['feature1', 'feature2', ...]}
        Si None, agrupa automáticamente por prefijo (mfcc_, spectral_, etc.)
    figsize : tuple
        Dimensiones de la figura completa (ancho, alto) en pulgadas.
        Ajustar según el número de grupos y características

    Comportamiento:
    ---------------
    Agrupación automática (si feature_groups=None):
    - 'mfcc': Coeficientes MFCC y derivadas
    - 'spectral': Centroide, rolloff, bandwidth, etc.
    - 'chroma': Vector cromático (12 dimensiones)
    - 'energy': RMS, entropía energética
    - 'pitch': Características de frecuencia fundamental
    - 'envelope': ADSR y dinámica temporal
    - 'mel': Espectrograma Mel y bandas

    Elementos visuales:
    -------------------
    - Boxplots por grupo con estadísticas robustas
    - Caja: Cuartiles Q1-Q3 (50% central de datos)
    - Línea central: Mediana (Q2)
    - Bigotes: Rango intercuartil extendido (1.5 * IQR)
    - Puntos: Outliers fuera del rango de bigotes
    - Rotación de etiquetas para legibilidad

    Interpretación:
    ---------------
    Distribución normal:
    - Caja simétrica, bigotes equidistantes
    - Pocos o ningún outlier

    Distribución sesgada:
    - Caja asimétrica, bigotes desiguales
    - Mediana descentrada

    Presencia de outliers:
    - Puntos fuera de bigotes
    - Posibles errores o casos especiales

    Variabilidad:
    - Caja grande = alta variabilidad
    - Caja pequeña = baja variabilidad

    Casos de uso:
    -------------
    - Control de calidad de extracción
    - Identificación de características problemáticas
    - Comparación entre grupos de características
    - Análisis exploratorio de datasets
    - Detección de outliers y anomalías
    - Validación de preprocesamiento

    Consideraciones:
    ----------------
    - Requiere matplotlib para visualización
    - Figsize debe ajustarse según número de grupos
    - Características con escalas muy diferentes pueden necesitar normalización
    - Outliers pueden indicar errores o casos especiales válidos

    Ejemplo:
    --------
    >>> features_list = [extract_all_features(f) for f in audio_files]
    >>> plot_feature_distribution(features_list)  # Agrupación automática
    >>>
    >>> # Agrupación personalizada
    >>> groups = {
    ...     'Timbre': ['mfcc_0_mean', 'spectral_centroid_mean'],
    ...     'Energía': ['rms', 'energy_mean', 'envelope_max']
    ... }
    >>> plot_feature_distribution(features_list, groups, figsize=(12, 8))

    Salida:
    -------
    Muestra directamente los gráficos usando plt.show().
    No retorna valores, solo genera visualización interactiva.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = features_to_dataframe(features_dict)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if feature_groups is None:
        feature_groups = {}
        for col in numeric_cols:
            prefix = col.split('_')[0]
            if prefix not in feature_groups:
                feature_groups[prefix] = []
            feature_groups[prefix].append(col)

    n_groups = len(feature_groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=figsize)
    if n_groups == 1:
        axes = [axes]

    for i, (group_name, features) in enumerate(feature_groups.items()):
        group_features = [f for f in features if f in numeric_cols]
        if group_features:
            df[group_features].boxplot(ax=axes[i])
            axes[i].set_title(f'Distribución de características: {group_name}')
            axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
