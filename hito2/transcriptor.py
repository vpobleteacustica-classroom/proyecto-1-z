import os
import subprocess
import sys
import onnxruntime as ort
import numpy as np
import soundfile as sf
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

def run_demucs(ruta_mp3, modelo="htdemucs_6s"):
    print(f"🎧 Separando pistas con {modelo}...")
    subprocess.run(["demucs", "-n", modelo, ruta_mp3], check=True)

    base_dir = os.path.join("separated", modelo)
    nombre = os.path.splitext(os.path.basename(ruta_mp3))[0]
    return os.path.join(base_dir, nombre)


def load_audio(path):
    data, sr = sf.read(path)
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = np.mean(data, axis=1)  # mezcla a mono
    return data.astype(np.float32), sr


def save_audio(path, data, sr):
    sf.write(path, data, sr)


def extract_guitar(input_path, model_path="mdx_guitar.onnx"):
    print("🎸 Aislando guitarra desde 'other.wav'...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo MDX Guitar no encontrado: {model_path}")

    audio, sr = load_audio(input_path)

    # Inicializa el modelo ONNX en GPU si disponible
    providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    # Entrada esperada: [batch, samples]
    audio = np.expand_dims(audio, axis=0)
    output = session.run(None, {"input": audio})[0].squeeze()

    output_path = os.path.splitext(input_path)[0] + "_guitar.wav"
    save_audio(output_path, output, sr)
    print(f"Guitarra aislada guardada en: {output_path}")
    return output_path


ruta_mp3 = "/song/atoutlemonde.mp3"
if not os.path.exists(ruta_mp3):
    print(f"Archivo no encontrado: {ruta_mp3}")
    sys.exit(1)

out_dir = run_demucs(ruta_mp3)
other_path = os.path.join(out_dir, "other.wav")

if not os.path.exists(other_path):
    print("No se encontró 'other.wav'; algo falló con Demucs.")


predict_and_save(
    ['/separated/htdemucs_6s/guitar.wav'],          # Nombre del archivo a convertir
    'transcription',                    # Carpeta de salida
    True,                               # save_midi
    False,                              # sonify_midi 
    False,                              # save_model_outputs
    True,                               # save_notes
    ICASSP_2022_MODEL_PATH              # model_or_model_path
)