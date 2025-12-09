import os
import numpy as np
import librosa
import soundfile as sf
import crepe
import pretty_midi
import math
from typing import List, Tuple, Optional
from scipy.signal import butter, sosfilt, medfilt

# Afinación estándar EADGBE
STRING_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
MAX_FRET = 20
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# ==========================================
# 1) EXTRACCIÓN DE MELODÍA 
# ==========================================

def trim_silence(y, top_db=30):
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trim

def bandpass_filter(y, sr, lowcut=80, highcut=3000):
    """Deja pasar solo frecuencias donde suele estar la nota fundamental de la voz."""
    sos = butter(10, [lowcut, highcut], btype='band', fs=sr, output='sos')
    return sosfilt(sos, y)

def extract_melody(vocals_path: str,
                   sr_target: int = 22050,
                   hop_length: int = 512,
                   crepe_step_ms: int = 40,
                   use_tiny: bool = True) -> List[Tuple[int, float, float]]:
    
    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"vocals not found: {vocals_path}")

    # 1. Cargar audio
    y, sr = librosa.load(vocals_path, sr=sr_target, mono=True)
    
    # 2. Filtro Pasa-Banda (85Hz - 3000Hz)
    sos = butter(10, [85, 3000], btype='band', fs=sr, output='sos')
    y = sosfilt(sos, y)

    # 3. Trim de silencio
    y = trim_silence(y, top_db=28)
    
    if len(y) == 0:
        return []

    # Parámetros crepe
    step_size = crepe_step_ms 
    model_capacity = 'tiny' if use_tiny else 'full'

    events = []
    block_sec = 60.0
    n_blocks = max(1, math.ceil(len(y) / (sr * block_sec)))
    
    for b in range(n_blocks):
        start_sample = int(b * block_sec * sr)
        end_sample = int(min(len(y), (b + 1) * block_sec * sr))
        y_block = y[start_sample:end_sample]

        # Predicción
        time_vals, frequency, confidence, _ = crepe.predict(
            y_block, sr, viterbi=True,
            model_capacity=model_capacity, step_size=step_size, verbose=0
        )

        freq = np.array(frequency)
        conf = np.array(confidence)
        
        # Umbral de confianza
        freq[conf < 0.30] = 0.0

        midi = np.zeros_like(freq, dtype=int)
        nonzero = freq > 0
        midi[nonzero] = np.round(69 + 12 * np.log2(freq[nonzero] / 440.0)).astype(int)
        
        # Filtro de Mediana
        midi = medfilt(midi, kernel_size=5).astype(int)

        # Convertir timestamps
        times = np.array(time_vals) + (start_sample / sr)

        # Agrupar eventos
        if midi.size == 0:
            continue
            
        cur_pitch = int(midi[0])
        cur_start = float(times[0])
        
        for i in range(1, len(midi)):
            if int(midi[i]) != cur_pitch:
                if cur_pitch != 0:
                    events.append((cur_pitch, cur_start, float(times[i])))
                cur_pitch = int(midi[i])
                cur_start = float(times[i])
        
        # Último evento del bloque
        if cur_pitch != 0:
            events.append((int(cur_pitch), cur_start, float(times[-1] + (step_size / 1000.0))))

    # Post-process: fusionar eventos muy cortos
    merged = []
    min_dur = 0.08
    for e in events:
        p, s, t = e
        if not merged:
            merged.append([p, s, t])
        else:
            last = merged[-1]
            # Si es la misma nota y el hueco es pequeño, unimos
            if last[0] == p and s - last[2] <= 0.1:
                last[2] = t
            else:
                merged.append([p, s, t])
    
    # ============================================
    # CORRECCIÓN Y NORMALIZACIÓN DE OCTAVA
    # ============================================
    def normalize_melody_to_single_octave(events, max_interval=7):
        """
        Mantiene toda la melodía en UNA SOLA octava coherente.
        
        Proceso:
        1. Detecta la octava más común (octava dominante)
        2. Fuerza TODAS las notas a esa octava
        3. Solo permite movimientos melódicos pequeños (≤ max_interval)
        
        Ejemplo:
        Entrada:  Do4, Re5, Mi4, Fa5, Sol4  (saltando entre octavas)
        Salida:   Do4, Re4, Mi4, Fa4, Sol4  (todo en octava 4)
        """
        if not events:
            return events
        
        # PASO 1: Identificar la octava dominante
        octaves = [pitch // 12 for pitch, _, _ in events]
        
        # Contar frecuencia de cada octava
        from collections import Counter
        octave_counts = Counter(octaves)
        
        # La octava más común es la "correcta"
        dominant_octave = octave_counts.most_common(1)[0][0]
        
        print(f"      → Octava dominante detectada: {dominant_octave} (C{dominant_octave} = MIDI {dominant_octave * 12})")
        
        # PASO 2: Normalizar todas las notas a la octava dominante
        normalized = []
        
        for pitch, start, end in events:
            # Extraer la clase de nota (Do=0, Re=2, Mi=4, etc.)
            pitch_class = pitch % 12
            
            # Reconstruir la nota en la octava dominante
            normalized_pitch = dominant_octave * 12 + pitch_class
            
            # Asegurar que esté en rango tocable (C3 a C6)
            while normalized_pitch < 48:
                normalized_pitch += 12
            while normalized_pitch > 84:
                normalized_pitch -= 12
            
            normalized.append([normalized_pitch, start, end])
        
        # PASO 3: Corrección de saltos melódicos imposibles
        corrected = [normalized[0]]
        
        for i in range(1, len(normalized)):
            prev_pitch = corrected[-1][0]
            curr_pitch = normalized[i][0]
            
            interval = abs(curr_pitch - prev_pitch)
            
            # Si aún hay un salto grande (> max_interval), es un error de detección
            if interval > max_interval:
                # Buscar la nota más cercana a la anterior (misma octava o adyacente)
                pitch_class = curr_pitch % 12
                
                candidates = []
                for octave_offset in [-1, 0, 1]:  # Octava anterior, actual, siguiente
                    candidate = (prev_pitch // 12 + octave_offset) * 12 + pitch_class
                    if 48 <= candidate <= 84:
                        distance = abs(candidate - prev_pitch)
                        candidates.append((distance, candidate))
                
                if candidates:
                    candidates.sort()
                    best_pitch = candidates[0][1]
                    corrected.append([best_pitch, normalized[i][1], normalized[i][2]])
                else:
                    corrected.append(normalized[i])
            else:
                # Salto melódico normal, mantener
                corrected.append(normalized[i])
        
        return corrected
    
    # Aplicar normalización de octava
    corrected = normalize_melody_to_single_octave(merged, max_interval=7)
    
    # Filtrar por duración mínima
    final = []
    for p, s, t in corrected:
        if t - s >= min_dur:
            final.append((int(p), float(s), float(t)))
            
    return final


# ==========================================
# 2) DETECCIÓN DE ACORDES
# ==========================================

def detect_chords_improved(audio_paths: List[str],
                          sr_target: int = 22050) -> List[Tuple[str, float, float, int]]:
    
    # Cargar y mezclar pistas armónicas
    mix = None
    for path in audio_paths:
        if not os.path.exists(path):
            continue
        y, _ = librosa.load(path, sr=sr_target, mono=True)
        if mix is None:
            mix = y
        else:
            if len(y) < len(mix):
                y = np.pad(y, (0, len(mix) - len(y)))
            elif len(y) > len(mix):
                mix = np.pad(mix, (0, len(y) - len(mix)))
            mix += y
    
    if mix is None:
        return []
    
    # Normalizar
    mix = mix / (np.max(np.abs(mix)) + 1e-8)
    
    hop_length = 2048
    chroma = librosa.feature.chroma_cqt(
        y=mix, 
        sr=sr_target, 
        hop_length=hop_length,
        n_chroma=12,
        bins_per_octave=36
    )
    
    # Detección de bajo para root
    bass_chroma = librosa.feature.chroma_cqt(
        y=mix,
        sr=sr_target,
        hop_length=hop_length,
        fmin=librosa.note_to_hz('E2'),
        n_chroma=12
    )
    
    # Templates de acordes expandidos
    chord_templates = {
        'maj': [1.0, 0, 0, 0, 0.6, 0, 0, 0.8, 0, 0, 0, 0],
        'min': [1.0, 0, 0, 0.6, 0, 0, 0, 0.8, 0, 0, 0, 0],
        '7': [1.0, 0, 0, 0, 0.5, 0, 0, 0.7, 0, 0, 0.6, 0],
        'sus4': [1.0, 0, 0, 0, 0, 0.8, 0, 0.6, 0, 0, 0, 0]
    }
    
    # Análisis por ventanas de ~1 segundo
    window_sec = 1.0
    frames_per_win = max(1, int(window_sec * sr_target / hop_length))
    num_windows = int(np.ceil(chroma.shape[1] / frames_per_win))
    
    chords = []
    for w in range(num_windows):
        start_frame = w * frames_per_win
        end_frame = min(chroma.shape[1], (w + 1) * frames_per_win)
        
        chroma_vec = np.mean(chroma[:, start_frame:end_frame], axis=1)
        bass_vec = np.mean(bass_chroma[:, start_frame:end_frame], axis=1)
        
        # Root más probable según el bajo
        root_from_bass = int(np.argmax(bass_vec))
        
        # Encontrar mejor acorde
        best_match = None
        best_score = 0.0
        
        for root in range(12):
            for quality, template in chord_templates.items():
                # Rotar template
                rotated = np.roll(template, root)
                score = np.dot(chroma_vec, rotated)
                
                # Bonus si el root coincide con el bajo
                if root == root_from_bass:
                    score *= 1.3
                
                if score > best_score:
                    best_score = score
                    best_match = (root, quality)
        
        t_start = start_frame * hop_length / sr_target
        t_end = end_frame * hop_length / sr_target
        
        if best_match:
            root, quality = best_match
            chord_name = f"{NOTE_NAMES[root]}{quality}"
            root_midi = 48 + root  # C3 como base
            chords.append((chord_name, float(t_start), float(t_end), root_midi))
    
    # Merge acordes consecutivos iguales
    merged = []
    for chord in chords:
        if not merged:
            merged.append(list(chord))
        else:
            prev = merged[-1]
            if prev[0] == chord[0] and abs(prev[2] - chord[1]) < 0.15:
                prev[2] = chord[2]
            else:
                merged.append(list(chord))
    
    return [tuple(c) for c in merged]


# ==========================================
# 3) VOICING CON VOICE LEADING
# ==========================================

class VoicingEngine:
    def __init__(self):
        self.last_voicing = None
        
    def get_voicing(self, root_midi: int, quality: str, 
                   prev_melody_pitch: Optional[int] = None) -> List[int]:
        
        # Ajustar root a registro medio-bajo (E2-A3)
        root = root_midi
        while root > 57:
            root -= 12
        while root < 40:
            root += 12
        
        # Construir triada básica
        if 'maj' in quality or quality == 'sus4':
            intervals = [0, 4, 7] if 'maj' in quality else [0, 5, 7]
        elif 'min' in quality:
            intervals = [0, 3, 7]
        elif '7' in quality:
            intervals = [0, 4, 7, 10]
        else:
            intervals = [0, 4, 7]
        
        # Generar voicing inicial
        notes = [root + i for i in intervals]
        
        # Drop 2 voicing (guitarra típica)
        if len(notes) >= 3:
            # Bajar segunda nota una octava
            notes = [notes[0], notes[2], notes[1] + 12]
            if len(notes) >= 4:
                notes.append(notes[3])
        
        # Si hay melodía, evitar el registro
        if prev_melody_pitch:
            notes = [n for n in notes if abs(n - prev_melody_pitch) > 4]
        
        # Voice leading: si hay voicing previo, minimizar movimiento
        if self.last_voicing:
            # Calcular distancia total
            def total_movement(voicing):
                return sum(abs(a - b) for a, b in zip(sorted(voicing), sorted(self.last_voicing)))
            
            # Probar inversiones
            best_voicing = notes
            best_distance = total_movement(notes)
            
            for inversion in range(1, len(notes)):
                inv = notes[inversion:] + [n + 12 for n in notes[:inversion]]
                dist = total_movement(inv)
                if dist < best_distance:
                    best_distance = dist
                    best_voicing = inv
            
            notes = best_voicing
        
        self.last_voicing = notes
        return notes


# ==========================================
# 4) CONSTRUCCIÓN DE ARREGLO
# ==========================================

def build_arrangement_v3(melody_events: List[Tuple[int,float,float]],
                        chords: List[Tuple[str,float,float,int]],
                        tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    guitar = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program('Acoustic Guitar (nylon)'),
        name="Guitar_Fingerstyle"
    )
    
    voicing_engine = VoicingEngine()
    
    # Calcular beat duration
    beat_duration = 60.0 / tempo
    
    # Generar acompañamiento
    for chord_name, start, end, root_midi in chords:
        duration = end - start
        
        # Determinar si hay melodía activa
        melody_notes = [p for p, s, e in melody_events if not (e <= start or s >= end)]
        has_melody = len(melody_notes) > 0
        avg_melody = int(np.mean(melody_notes)) if melody_notes else None
        
        # Obtener voicing
        quality = chord_name[len(NOTE_NAMES[root_midi % 12]):]  # extraer quality
        voicing = voicing_engine.get_voicing(root_midi, quality, avg_melody)
        
        # Dinámica adaptativa
        base_velocity = 55 if has_melody else 68
        
        # Patrón fingerstyle: bajo + arpeggio
        if duration >= beat_duration * 2:
            # Bajo en downbeat
            bass = pretty_midi.Note(
                velocity=base_velocity + 10,
                pitch=voicing[0],
                start=start,
                end=start + min(beat_duration * 0.7, duration * 0.4)
            )
            guitar.notes.append(bass)
            
            # Arpeggio de las otras notas
            step = beat_duration / 2
            for i, pitch in enumerate(voicing[1:]):
                note_start = start + step * (i + 1)
                if note_start >= end:
                    break
                note = pretty_midi.Note(
                    velocity=base_velocity,
                    pitch=pitch,
                    start=note_start,
                    end=min(end, note_start + step * 1.5)
                )
                guitar.notes.append(note)
        else:
            # Acordes cortos: strum simple
            for pitch in voicing:
                note = pretty_midi.Note(
                    velocity=base_velocity,
                    pitch=pitch,
                    start=start,
                    end=end
                )
                guitar.notes.append(note)
    
    # Añadir melodía con prioridad
    for pitch, start, end in melody_events:
        # Ajustar octava si es necesario
        p = pitch
        if p < 60:
            p += 12
        elif p > 84:
            p -= 12
        
        # Velocidad fija alta
        velocity = 105
        
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=p,
            start=start,
            end=end
        )
        guitar.notes.append(note)
    
    pm.instruments.append(guitar)
    return pm


# ==========================================
# 5) TABLATURA OPTIMIZADA
# ==========================================

def optimize_fingering(notes_with_time: List[Tuple[int, float, float]]) -> List[Tuple[int, int, float, float]]:

    result = []
    last_position = 0  # Posición media del traste anterior
    
    for pitch, start, end in notes_with_time:
        # Encontrar todas las opciones posibles
        options = []
        for string_idx in range(6):
            tuning = STRING_TUNING[string_idx]
            fret = pitch - tuning
            if 0 <= fret <= MAX_FRET:
                # Penalizar según distancia de posición previa
                distance_penalty = abs(fret - last_position)
                # Penalizar cuerdas graves para melodía
                string_penalty = (5 - string_idx) * 2
                score = distance_penalty + string_penalty
                options.append((score, string_idx, fret))
        
        if options:
            options.sort()
            _, string_idx, fret = options[0]
            result.append((string_idx, fret, start, end))
            last_position = fret
    
    return result


def midi_to_tab_improved(pm: pretty_midi.PrettyMIDI, step: float = 0.15) -> str:
    inst = pm.instruments[0]
    
    # Extraer y optimizar fingering
    notes = [(n.pitch, n.start, n.end) for n in inst.notes]
    fingering = optimize_fingering(sorted(notes, key=lambda x: x[1]))
    
    if not fingering:
        return "No notes to display"
    
    # Crear grid temporal
    start_time = min(s for _, _, s, _ in fingering)
    end_time = max(e for _, _, _, e in fingering)
    
    num_cols = int((end_time - start_time) / step) + 1
    tab = [['--' for _ in range(num_cols)] for _ in range(6)]
    
    for string_idx, fret, start, end in fingering:
        col = int((start - start_time) / step)
        if 0 <= col < num_cols:
            tab[string_idx][col] = f"{fret:02d}"
    
    # Formatear salida
    string_names = ['e', 'B', 'G', 'D', 'A', 'E']
    lines = []
    for i in range(6):
        line = "|".join(tab[5-i])
        lines.append(f"{string_names[i]}|{line}|")
    
    return "\n".join(lines)


# ==========================================
# FUNCIONES AUXILIARES PARA GUARDAR MELODÍA Y ACORDES
# ==========================================

def save_melody_to_midi(melody_events: List[Tuple[int, float, float]], output_path: str):
    pm = pretty_midi.PrettyMIDI()
    melody_inst = pretty_midi.Instrument(program=0, name="Vocal Melody")
    
    for pitch, start, end in melody_events:
        note = pretty_midi.Note(    
            velocity=100,
            pitch=int(pitch),
            start=float(start),
            end=float(end)
        )
        melody_inst.notes.append(note)
    
    pm.instruments.append(melody_inst)
    pm.write(output_path)
    return pm


def save_chords_to_midi(chords_list: List[Tuple[str, float, float, int]], output_path: str):
    pm = pretty_midi.PrettyMIDI()
    chord_inst = pretty_midi.Instrument(program=4, name="Harmony_Chords")

    for chord_name, start, end, root_midi in chords_list:
        # Extraer quality del nombre del acorde
        root_idx = root_midi % 12
        quality = chord_name[len(NOTE_NAMES[root_idx]):]
        
        # Construir triada
        notes_in_chord = [root_midi]
        
        if 'maj' in quality or quality == '':
            notes_in_chord.append(root_midi + 4)
            notes_in_chord.append(root_midi + 7)
        elif 'min' in quality:
            notes_in_chord.append(root_midi + 3)
            notes_in_chord.append(root_midi + 7)
        elif '7' in quality:
            notes_in_chord.append(root_midi + 4)
            notes_in_chord.append(root_midi + 7)
            notes_in_chord.append(root_midi + 10)
        else:
            notes_in_chord.append(root_midi + 4)
            notes_in_chord.append(root_midi + 7)

        # Crear notas MIDI
        for note_pitch in notes_in_chord:
            note = pretty_midi.Note(
                velocity=65,
                pitch=int(note_pitch),
                start=float(start),
                end=float(end)
            )
            chord_inst.notes.append(note)

    pm.instruments.append(chord_inst)
    pm.write(output_path)
    return pm


# ==========================================
# 6) PIPELINE PRINCIPAL
# ==========================================

def run_pipeline(song_dir: str, out_dir: str = "arrangement_v3", max_tempo: int = 140):
    """
    Args:
        song_dir: Directorio con archivos separados
        out_dir: Directorio de salida
        max_tempo: Tempo máximo permitido (BPM). Si la canción es más rápida, se reduce.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    vocals = os.path.join(song_dir, "vocals.wav")
    guitar = os.path.join(song_dir, "guitar.wav")
    piano = os.path.join(song_dir, "piano.wav")
    bass = os.path.join(song_dir, "bass.wav")
    other = os.path.join(song_dir, "other.wav")
    
    print("=" * 60)
    print("PIPELINE: Arreglo de Guitarra")
    print("=" * 60)
    
    # 0. DETECTAR TEMPO REAL DE LA CANCIÓN
    print("\n[0/4] Detectando tempo de la canción...")
    
    # Cargar una pista para detectar tempo (preferir bajo o batería si existe)
    tempo_source = None
    for path in [bass, other, guitar, vocals]:
        if os.path.exists(path):
            tempo_source = path
            break
    
    if tempo_source:
        y_tempo, sr_tempo = librosa.load(tempo_source, sr=22050, duration=60)  # Solo primeros 60s
        onset_env = librosa.onset.onset_strength(y=y_tempo, sr=sr_tempo)
        detected_tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr_tempo)[0]
        
        print(f"      ✓ Tempo detectado: {detected_tempo:.1f} BPM")
        
        # Aplicar límite de tempo
        if detected_tempo > max_tempo:
            final_tempo = max_tempo
            reduction_percent = ((detected_tempo - max_tempo) / detected_tempo) * 100
            print(f"      ⚠ Tempo muy rápido! Reduciendo de {detected_tempo:.1f} a {final_tempo} BPM")
            print(f"        (Reducción del {reduction_percent:.1f}%)")
        else:
            final_tempo = detected_tempo
            print(f"      ✓ Tempo adecuado: {final_tempo:.1f} BPM")
    else:
        final_tempo = 120.0
        print(f"      ⚠ No se pudo detectar tempo, usando {final_tempo} BPM por defecto")
    
    # 1. Melodía (TU VERSIÓN ORIGINAL)
    print("\n[1/4] Extrayendo melodía (CREPE tiny + filtros)...")
    melody = extract_melody(vocals, crepe_step_ms=40, hop_length=512, use_tiny=True)
    print(f"      ✓ {len(melody)} eventos melódicos detectados")
    
    melody_mid = os.path.join(out_dir, "melody.mid")
    save_melody_to_midi(melody, melody_mid)
    print(f"      📄 MIDI melodía: {melody_mid}")
    
    # 2. Acordes 
    print("\n[2/4] Detectando acordes (análisis armónico)...")
    chords = detect_chords_improved([guitar, piano, bass, other])
    print(f"      ✓ {len(chords)} cambios de acorde")
    if chords:
        print(f"      Ejemplo: {[c[0] for c in chords[:3]]}")
    
    chord_mid = os.path.join(out_dir, "chords.mid")
    save_chords_to_midi(chords, chord_mid)
    print(f"      📄 MIDI acordes: {chord_mid}")
    
    # 3. Arreglo (con el tempo ajustado)
    print(f"\n[3/4] Generando arreglo a {final_tempo:.1f} BPM...")
    pm = build_arrangement_v3(melody, chords, tempo=final_tempo)
    midi_path = os.path.join(out_dir, "arrangement.mid")
    pm.write(midi_path)
    print(f"      ✓ MIDI guardado: {midi_path}")
    
    # 4. Tablatura (optimizada)
    print("\n[4/4] Creando tablatura optimizada...")
    tab = midi_to_tab_improved(pm, step=0.15)
    tab_path = os.path.join(out_dir, "tablatura.txt")
    with open(tab_path, "w", encoding="utf-8") as f:
        f.write(tab)
    print(f"      ✓ TAB guardada: {tab_path}")
    
    print("\n" + "=" * 60)
    print("✓ Pipeline completado exitosamente")
    print("=" * 60)
    print(f"\n📊 Resumen:")
    print(f"   • Tempo final: {final_tempo:.1f} BPM")
    print(f"   • Notas melódicas: {len(melody)}")
    print(f"   • Cambios de acorde: {len(chords)}")
    
    return {
        "midi": midi_path,
        "tab": tab_path,
        "melody_midi": melody_mid,
        "chords_midi": chord_mid,
        "tempo": final_tempo,
        "melody_events": len(melody),
        "chord_changes": len(chords)
    }


if __name__ == "__main__":
    song_folder = "separated/htdemucs_6s/judas30"
    
    # Ajustar max_tempo según preferencia:
    # 100 = Muy lento (baladas)
    # 120 = Moderado (pop)
    # 140 = Rápido (rock)
    # 160 = Muy rápido (punk/metal)
    
    result = run_pipeline(
        song_folder,
        max_tempo=140
    )
    
    print("\n🎸 Archivos generados:")
    for key, value in result.items():
        print(f"   {key}: {value}")