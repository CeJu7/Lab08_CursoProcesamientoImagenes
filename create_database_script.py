from labs.feature_extraction import extract_all_features, features_to_dataframe
from pathlib import Path
from datetime import datetime


def process_audio_files(dataset_path='data/procesados', output_dir='results'):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    audio_files = list(dataset_path.rglob('*.wav'))
    print(f"Procesando {len(audio_files)} archivos...")

    features_list = []
    processed = 0

    for i, audio_path in enumerate(audio_files):
        try:
            if i % max(1, len(audio_files) // 10) == 0:
                progress = (i + 1) / len(audio_files) * 100
                print(f"\rProgreso: {progress:.0f}%", end="", flush=True)

            features = extract_all_features(
                filename=str(audio_path),
                sr=22050,
                normalize=True,
                remove_silence=False
            )

            parts = audio_path.stem.split('_')
            individual_id = parts[0] if len(parts) >= 3 else audio_path.parent.parent.name
            audio_type = parts[1] if len(parts) >= 3 else audio_path.parent.name

            # Metadatos
            features['individual_id'] = individual_id
            features['audio_type'] = audio_type
            features['label'] = 1 if audio_type == 'fake' else 0

            features_list.append(features)
            processed += 1

        except Exception:
            continue

    print(f"\nProcesados: {processed}/{len(audio_files)}")

    if not features_list:
        print("No se procesaron archivos")
        return

    df = features_to_dataframe(features_list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = output_dir / f"fake_audio_features_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Guardado: {csv_file}")


if __name__ == "__main__":
    process_audio_files()
