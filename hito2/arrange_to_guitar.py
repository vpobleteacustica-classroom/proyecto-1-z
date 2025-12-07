# arrange_to_guitar_v2.py
import os
import numpy as np
import librosa
import soundfile as sf
import crepe
import pretty_midi
import math
from typing import List, Tuple
from scipy.signal import butter, sosfilt, medfilt

# Afinación estándar (MIDI numbers) para las 6 cuerdas E A D G B E (cuerda baja = 0 index)
STRING_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
MAX_FRET = 20

# --------------------------
# 1) EXTRACCIÓN DE MELODÍA (CREPE rápido y por bloques)
# --------------------------

def trim_silence(y, top_db=30):
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trim

def bandpass_filter(y, sr, lowcut=80, highcut=3000):
    """Deja pasar solo frecuencias donde suele estar la nota fundamental de la voz."""
    sos = butter(10, [lowcut, highcut], btype='band', fs=sr, output='sos')
    return sosfilt(sos, y)

from scipy.signal import butter, sosfilt, medfilt

def extract_melody(vocals_path: str,
                   sr_target: int = 22050,
                   hop_length: int = 512,
                   crepe_step_ms: int = 40,
                   use_tiny: bool = True) -> List[Tuple[int, float, float]]:
    
    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"vocals not found: {vocals_path}")

    # 1. Cargar audio
    y, sr = librosa.load(vocals_path, sr=sr_target, mono=True)
    
    # 2. MEJORA: Filtro Pasa-Banda (85Hz - 3000Hz)
    # Elimina graves sucios y agudos sin tono
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
            y_block, sr, viterbi=True, # MEJORA: viterbi=True suaviza la ruta internamente
            model_capacity=model_capacity, step_size=step_size, verbose=0
        )

        freq = np.array(frequency)
        conf = np.array(confidence)
        
        # Umbral de confianza
        freq[conf < 0.30] = 0.0 # Subí un poco el umbral a 0.30 para ser más estricto

        midi = np.zeros_like(freq, dtype=int)
        nonzero = freq > 0
        midi[nonzero] = np.round(69 + 12 * np.log2(freq[nonzero] / 440.0)).astype(int)
        
        # 4. MEJORA: Filtro de Mediana para eliminar "jitter"
        # Elimina picos de 1 frame de duración que suenan mal
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
    min_dur = 0.08 # Subí un poco la duración mínima
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
                
    final = []
    for p,s,t in merged:
        if t - s >= min_dur:
            final.append((int(p), float(s), float(t)))
            
    return final
# --------------------------
# 2) ESTIMAR ACORDES (Chroma sencillo)
# --------------------------
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def chroma_chords(audio_paths: List[str],
                  sr_target: int = 22050,
                  hop_length: int = 2048,
                  window_sec: float = 0.5) -> List[Tuple[int,str,float,float]]:
    mix = None
    for p in audio_paths:
        if not os.path.exists(p):
            continue
        y, sr = librosa.load(p, sr=sr_target, mono=True)
        if mix is None:
            mix = y
        else:
            if len(y) < len(mix):
                y = np.pad(y, (0, len(mix)-len(y)))
            elif len(mix) < len(y):
                mix = np.pad(mix, (0, len(y)-len(mix)))
            mix += y
    if mix is None:
        return []
    chroma = librosa.feature.chroma_cqt(y=mix, sr=sr_target, hop_length=hop_length)
    frames_per_win = max(1, int(window_sec * sr_target / hop_length))
    num_windows = int(np.ceil(chroma.shape[1] / frames_per_win))
    chords = []
    for w in range(num_windows):
        start = w * frames_per_win
        end = min(chroma.shape[1], (w+1) * frames_per_win)
        vec = np.mean(chroma[:, start:end], axis=1)
        best = None
        best_score = 0.0
        for root in range(12):
            maj_idxs = [(root + 0) % 12, (root + 4) % 12, (root + 7) % 12]
            min_idxs = [(root + 0) % 12, (root + 3) % 12, (root + 7) % 12]
            maj_score = vec[maj_idxs].sum()
            min_score = vec[min_idxs].sum()
            if maj_score > best_score:
                best_score = maj_score
                best = (root, 'maj')
            if min_score > best_score:
                best_score = min_score
                best = (root, 'min')
        t_start = start * hop_length / sr_target
        t_end = end * hop_length / sr_target
        if best is not None:
            chords.append((best[0], best[1], float(t_start), float(t_end)))
    # merge
    merged = []
    for c in chords:
        if not merged:
            merged.append(c)
        else:
            prev = merged[-1]
            if prev[0] == c[0] and prev[1] == c[1] and abs(prev[3]-c[2]) < 0.12:
                merged[-1] = (prev[0], prev[1], prev[2], c[3])
            else:
                merged.append(c)
    return merged

# --------------------------
# Util: mapear pitch a string+fret (mejor greedy)
# --------------------------
def pitch_to_string_fret(pitch: int) -> Tuple[int,int]:
    best = None
    best_fret = 999
    for string_idx in range(6):  # 0=lowE
        tuning = STRING_TUNING[string_idx]
        fret = int(round(pitch - tuning))
        if 0 <= fret <= MAX_FRET:
            # preferimos cuerdas más agudas si el traste es similar (para melodía)
            score = fret + (string_idx * 0.1)
            if fret < best_fret:
                best_fret = fret
                best = (string_idx, fret)
    if best is None:
        for octave_shift in [-12, 12]:
            new = pitch + octave_shift
            for string_idx in range(6):
                tuning = STRING_TUNING[string_idx]
                fret = int(round(new - tuning))
                if 0 <= fret <= MAX_FRET:
                    return (string_idx, fret)
    return best

# --------------------------
# 3) VOICINGS REALISTAS y PATRÓN POP (build_arrangement_v2)
# --------------------------
def choose_voicing_spread(root_midi: int, ctype: str):
    """
    Crea un voicing con spread (notas separadas por octavas) para sonar claro en guitarra.
    root_midi: any pitch class MIDI (we'll adjust to playable region)
    """
    # poner root alrededor de 40-55 para tener bajo presente
    base = root_midi
    while base > 55:
        base -= 12
    while base < 40:
        base += 12
    if ctype == 'maj':
        third = base + 4
    else:
        third = base + 3
    fifth = base + 7
    # Spread: si las notas están muy juntas, mover la tercera una octava arriba
    if (third - base) < 5:
        third += 12
    if (fifth - third) < 4:
        fifth += 12
    # devolver listado (bajo, mid, high)
    return [base, third, fifth]

def build_arrangement_v2(melody_events: List[Tuple[int,float,float]],
                         chords: List[Tuple[int,str,float,float]],
                         style: str = 'pop') -> pretty_midi.PrettyMIDI:
    """
    Crea un arreglo más natural:
    - Para cada acorde genera un pattern: bajo en beat + 3 notas arpegiadas (pop fingerstyle)
    - Cuando la melodía coincide, reduce la intensidad del acompañamiento
    """
    pm = pretty_midi.PrettyMIDI()
    guitar_program = pretty_midi.instrument_name_to_program('Acoustic Guitar (nylon)')
    guitar_inst = pretty_midi.Instrument(program=guitar_program, name="Guitar_arr")

    # Index melody by time for quick lookup
    melody_index = []
    for p,s,e in melody_events:
        melody_index.append((p,s,e))
    # función helper para saber si hay melodía en intervalo
    def melody_active(t0,t1):
        for _,s,e in melody_index:
            if not (e <= t0 or s >= t1):
                return True
        return False

    # crear acordes como patrones
    for c in chords:
        root, ctype, s, e = c
        # obtener root pitch en MIDI (ej: root representa 0=C)
        # mapear root a un MIDI cercano (C4=60). Aquí buscamos la nota del root cercana a C4.
        root_midi = 60 + (root - (60 % 12))
        # mover root hacia registro tocable
        while root_midi - 12 >= 40 and (root_midi - 12) >= 40:
            root_midi -= 12
        voicing = choose_voicing_spread(root_midi, ctype)  # [low, mid, high]
        dur = max(0.5, e - s)
        # pattern pop: bass on downbeat, then arpeggio notes spaced
        # dividir en 4 golpes por compás (aprox)
        # offsets relativos (en segundos) - adaptativos según dur
        if dur < 0.8:
            offsets = [0.0, 0.15, 0.30]
        else:
            offsets = [0.0, dur*0.25, dur*0.5]
        # si hay melodía activa en la ventana, haremos el acompañamiento más suave
        is_mel = melody_active(s, e)
        chord_velocity = 70 if not is_mel else 54
        bass_note = voicing[0]
        # añadir bajo (nota más corta pero clara)
        bass = pretty_midi.Note(velocity=chord_velocity+5, pitch=int(bass_note), start=float(s), end=float(min(e, s + dur*0.6)))
        guitar_inst.notes.append(bass)
        # arpegiar las otras notas (mid, high)
        for i, off in enumerate(offsets):
            t_on = s + off
            if i == 0:
                pitch = voicing[1]
            else:
                pitch = voicing[2] if i==1 else voicing[1]
            # ajustar rango si necesario
            p = int(pitch)
            if p < 40:
                p += 12
            if p > 88:
                p -= 12
            end_time = float(min(e, t_on + dur*0.9))

            # Evitar notas inválidas
            if end_time <= t_on + 1e-4:
                continue  # saltar nota problemática

            note = pretty_midi.Note(
                velocity=chord_velocity,
                pitch=p,
                start=float(t_on),
                end=end_time
            )
            guitar_inst.notes.append(note)

    # añadir melodía: preferir registro alto, y darle prioridad dinámica
    for mel in melody_events:
        p, s, e = mel
        pitch = int(p)
        # preferir registro alto: si pitch <= 64, subir octava
        if pitch <= 64:
            pitch += 12
        # limitar rango
        if pitch > 88:
            pitch -= 12
        note = pretty_midi.Note(velocity=110, pitch=pitch, start=float(s), end=float(e))
        guitar_inst.notes.append(note)

    pm.instruments.append(guitar_inst)
    return pm

# --------------------------
# 4) TAB: generar desde notas + pitch_to_string_fret (mejorado)
# --------------------------
def midi_to_tab_text(pm: pretty_midi.PrettyMIDI, step: float = 0.25) -> List[str]:
    tab_lines = [[] for _ in range(6)]
    inst = pm.instruments[0]
    times = []
    for n in inst.notes:
        times.append((float(n.start), float(n.end), int(n.pitch)))
    times.sort()
    if not times:
        return ["No notes detected"]
    start_all = times[0][0]
    end_all = times[-1][1]
    t = start_all
    while t < end_all + 1e-9:
        slice_notes = [p for (s,e,p) in times if s <= t < e]
        col = ['--'] * 6
        # si hay multiples notas, preferir asignar melodía en cuerdas agudas
        for p in slice_notes:
            mapping = pitch_to_string_fret(p)
            if mapping is not None:
                s_idx, fret = mapping
                # priorizamos que melodía quede en cejilla alta cuando sea posible
                if col[s_idx] == '--':
                    col[s_idx] = f"{fret:02d}"
                else:
                    # si ya hay nota, dejar ambas: concatenar con comma (simplificado)
                    col[s_idx] = col[s_idx] + ',' + f"{fret:02d}"
        for i in range(6):
            tab_lines[i].append(col[i])
        t += step
    text_lines = []
    string_names = ['E','A','D','G','B','e']
    for i in range(5, -1, -1):
        line = "|".join(tab_lines[i])
        text_lines.append(f"{string_names[i]}|{line}|")
    return text_lines

def save_melody_to_midi(melody_events: List[Tuple[int, float, float]], output_path: str):
    """
    Recibe la lista [(pitch, start, end)...] y guarda un archivo .mid
    usando pretty_midi.
    """
    # 1. Crear objeto PrettyMIDI
    pm = pretty_midi.PrettyMIDI()

    # 2. Crear un Instrumento (Program 0 = Acoustic Grand Piano)
    # Puedes cambiar el program a 40 (Violin) o 73 (Flute) si prefieres
    melody_inst = pretty_midi.Instrument(program=0, name="Vocal Melody")

    # 3. Convertir cada evento de tu lista en una Nota de pretty_midi
    for pitch, start, end in melody_events:
        note = pretty_midi.Note(    
            velocity=100,      # Volumen fuerte (0-127)
            pitch=int(pitch),  # Asegurarse que es entero
            start=float(start),
            end=float(end)
        )
        melody_inst.notes.append(note)

    # 4. Añadir el instrumento al objeto MIDI y guardar
    pm.instruments.append(melody_inst)
    pm.write(output_path)
    return pm # Opcional: retornamos el objeto por si quieres seguir usándolo en memoria


def save_chords_to_midi(chords_list: List[Tuple[int, str, float, float]], output_path: str):
    """
    Convierte la lista de acordes [(root, 'maj'/'min', start, end)...] 
    en un archivo MIDI polifónico (triadas simples).
    """
    pm = pretty_midi.PrettyMIDI()
    # Usamos un piano eléctrico o Pad para que suene bien la armonía
    # Program 4 = Electric Piano 1, Program 89 = Warm Pad
    chord_inst = pretty_midi.Instrument(program=4, name="Harmony_Chords")

    for root_idx, quality, start, end in chords_list:
        # 1. Mapear el root (0-11) a una octava central (ej. C4 = 60)
        # root_idx 0 es C. 60 es C4.
        root_midi = 60 + root_idx 
        
        # 2. Construir la triada (Acorde básico de 3 notas)
        notes_in_chord = [root_midi] # La fundamental
        
        if quality == 'maj':
            notes_in_chord.append(root_midi + 4) # Tercera Mayor
            notes_in_chord.append(root_midi + 7) # Quinta Justa
        else: # 'min'
            notes_in_chord.append(root_midi + 3) # Tercera Menor
            notes_in_chord.append(root_midi + 7) # Quinta Justa

        # 3. Crear las notas MIDI para cada componente del acorde
        for note_pitch in notes_in_chord:
            note = pretty_midi.Note(
                velocity=65,       # Volumen medio (acompañamiento)
                pitch=int(note_pitch),
                start=float(start),
                end=float(end)
            )
            chord_inst.notes.append(note)

    pm.instruments.append(chord_inst)
    pm.write(output_path)
    return pm
# --------------------------
# 5) RUN PIPELINE (principal)
# --------------------------
def run_pipeline(song_dir: str, out_dir: str = "arrangement_out"):
    os.makedirs(out_dir, exist_ok=True)
    vocals = os.path.join(song_dir, "vocals.wav")
    guitar = os.path.join(song_dir, "guitar.wav")
    piano = os.path.join(song_dir, "piano.wav")
    other = os.path.join(song_dir, "other.wav")

    print("1) Extrayendo melodía (CREPE tiny + trimming)...")
    melody = extract_melody(vocals, crepe_step_ms=40, hop_length=512, use_tiny=True)

    melody_midi_path = os.path.join(out_dir, "melodia_vocals.mid")
    save_melody_to_midi(melody, melody_midi_path)
    print(f"    [OK] Melodía extraída guardada en: {melody_midi_path}")

    print(f"    Eventos de melodía: {len(melody)}")

    print("2) Estimando acordes (chroma CQT)...")
    chords = chroma_chords([guitar, piano, other], window_sec=0.6)
    print(f"    Acordes detectados: {len(chords)}")

    acordes_midi_path = os.path.join(out_dir, "acordes.mid")
    save_chords_to_midi(chords, acordes_midi_path)

    print("3) Construyendo arreglo (voicings realistas + patrón pop)...")
    pm = build_arrangement_v2(melody, chords, style='rock')
    midi_path = os.path.join(out_dir, "guitar_arrangement_v2.mid")
    pm.write(midi_path)
    print(f"    MIDI guardado en: {midi_path}")

    print("4) Generando tablatura textual...")
    tab = midi_to_tab_text(pm, step=0.25)
    tab_path = os.path.join(out_dir, "guitar_tab_v2.txt")
    with open(tab_path, "w") as f:
        f.write("\n".join(tab))
    print(f"    Tab guardada en: {tab_path}")

    return {"midi": midi_path, "tab": tab_path, "pretty_midi": pm}

if __name__ == "__main__":
    song_folder = "separated/htdemucs_6s/judas30"
    out = run_pipeline(song_folder)
    print("Listo.", out)
