# 🎶 Sistema de transcripción inteligente de guitarra a partir de grabaciones polifónicas

## 🧩 Descripción general
Este proyecto busca transcribir automáticamente pistas de guitarra desde grabaciones polifónicas completas. Utiliza técnicas de separación de fuentes de audio mediante redes neuronales preentrenadas (Demucs) y transcripción a formato MIDI (Basic Pitch de Spotify).

El sistema toma un archivo de audio, separa los instrumentos clave (bajo, batería, guitarra, piano, voz y otros), extrae la pista de guitarra y la convierte en un archivo MIDI de partituras.

---

## 🚀 Avances realizados
Se desarrolló una versión preliminar que:
- **Separa instrumentos** usando **Demucs** (modelo preentrenado htdemucs_6s) en 6 stems: bajo, batería, guitarra, piano, voz y otros
- **Aísla la guitarra** utilizando el modelo MDX Guitar para mayor precisión
- **Genera partituras MIDI** mediante **Basic Pitch** (laboratorio de audio inteligente de Spotify)
- Procesa audio completo de forma automática con pipeline integrado

---

## 🧱 Falta completar
El sistema cumple con los objetivos principales de separación y transcripción, sin embargo:
- El modelo de transcripción (Basic Pitch) tiene limitaciones en precisión con acordes complejos
- **Próximo hito:** Mejorar la calidad de las partituras generadas mediante:
  - Fine-tuning de modelos de transcripción
  - Post-procesamiento de notas MIDI
  - Validación con datasets de guitarra profesional
- **Objetivo futuro:** Adaptar la canción completa en una versión que mezcle las melodías principales, manteniendo coherencia y esencia original

---

## ⚙️ Cómo ejecutar el códigoPitch) tiene limitaciones en precisión con acordes complejos
- **Próximo hito:** Mejorar la calidad de las partituras generadas mediante:
  - Fine-tuning de modelos de transcripción
  - Post-procesamiento de notas MIDI
  - Validación con datasets de guitarra profesional
- **Objetivo futuro:** Adaptar la canción completa en una versión que mezcle las melodías principales, manteniendo coherencia y esencia original


### Opción 1: Ejecución Local

#### Requisitos del Sistema
- **Python 3.11** (requerido - no funciona con Python 3.12+)
- **FFmpeg** instalado en el sistema
- **GPU con CUDA** (opcional, mejora significativamente el rendimiento)
- Modelo `mdx_guitar.onnx` (descargar por separado)

#### Instalación

**1. Crear entorno virtual con Python 3.11**
```bash
# Con conda (recomendado)
conda create -n transcriptor python=3.11
conda activate transcriptor

# O con venv si tienes Python 3.11 instalado
python3.11 -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate  # En Windows
```

**2. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**3. Instalar FFmpeg**

```bash
  sudo apt-get install ffmpeg
```
**4. Descargar modelo MDX Guitar**

Descargar `mdx_guitar.onnx` y colocarlo en la carpeta raíz del proyecto.

#### Uso
```bash
python transcriptor.py
```

El script procesará el archivo de audio especificado y generará:
- Pistas separadas en `separated/htdemucs_6s/`
- Guitarra aislada con sufijo `_guitar.wav`
- Archivo MIDI de transcripción en carpeta `transcription/`

---

### Opción 2: Google Colab (Recomendado)
Para este proyecto, la ejecución se realiza actualmente en el entorno de Google Colab. Esta decisión se tomó debido a la complejidad en la gestión de las dependencias y las versiones de las librerías especializadas utilizadas. Al ejecutar el código en Colab, garantizamos un entorno de trabajo preconfigurado y estable, lo que minimiza los errores de compatibilidad entre las distintas versiones de las librerías de Python.

**Abrir el notebook en Google Colab**  
👉 [Abrir en Colab](https://colab.research.google.com/drive/1fGwE7eM9pQmY53bXG3p7JXR7huHOqKXb?usp=sharing)

---

## 📦 Dependencias principales

- **Demucs 4.0+** - Separación de fuentes de audio
- **Basic Pitch 0.2.5** - Transcripción audio a MIDI
- **PyTorch 2.0+** - Framework de deep learning
- **ONNX Runtime** - Inferencia de modelos MDX
- **TensorFlow 2.13** - Backend de Basic Pitch
- **NumPy, SoundFile, Librosa** - Procesamiento de audio

---

## ⚠️ Nota sobre compatibilidad

Este proyecto requiere **Python 3.11** debido a limitaciones de compatibilidad entre `basic-pitch`, `tensorflow` y versiones más recientes de Python. Una vez resueltos los conflictos de versiones en futuras actualizaciones de las librerías, se facilitará la ejecución con versiones más recientes de Python.

---

## 📝 Estructura del proyecto
```
proyecto/
├── transcriptor.py          # Script principal
├── requirements.txt         # Dependencias de Python
├── README.md               # Este archivo
├── mdx_guitar.onnx         # Modelo de extracción de guitarra
├── separated/              # Carpeta de salida (generada automáticamente)
└── transcription/          # Archivos MIDI generados
```

---

## 👥 Autores
Rudy Richter
Samantha Espinoza
Martín Maza
Esperanza Tejeda
