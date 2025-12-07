from mido import bpm2tempo, Message, MidiFile, MidiTrack, MetaMessage

import math


def map_to(midi_note, min_pitch, max_pitch):

    okt = int((max_pitch-min_pitch) // 12)
    if okt<=1:
        okt=1

    oktave=okt+1   
    macro_octave = midi_note // 4 # 0 to 35 (144/4 = 36)
      
    return (midi_note // oktave) + (min_pitch-6) 


def tokens_to_midi(tokens, reverse_vocab, output_path="output.mid", speed_factor=1.5, min_pitch=48, max_pitch=72, bpm=130, note_length=100):
    """
    Wandelt Token-Liste in MIDI-Datei um und beschleunigt die Wiedergabe.
    
    tokens         : Liste der Token-IDs
    reverse_vocab  : Mapping von Token-ID zu Token-String
    output_path    : Pfad für die MIDI-Datei
    speed_factor   : Faktor >1 = schneller, <1 = langsamer
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # BPM einstellen
    #bpm = 140  # gewünschte Geschwindigkeit
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm)))   


    current_time = 0
    active_notes = {}  # dict: note -> start_time
    ticks_per_beat = mid.ticks_per_beat

    for token_id in tokens:
        token_str = reverse_vocab[token_id]

        if token_str.startswith("TIME_"):
            delta = int(token_str.split("_")[1])
            # Geschwindigkeit anpassen
            delta = max(1, int(delta / speed_factor))
            current_time += delta
 
            if current_time <=0.5:
                continue

            if current_time >note_length:
                current_time=int(note_length)
                 

        elif token_str.startswith("NOTE_ON_"):
            pitch = int(token_str.split("_")[-1])

            pitch=map_to(pitch, min_pitch, max_pitch)

            velocity = 64  # Standard-Velocity
            active_notes[pitch] = current_time
            track.append(Message('note_on', note=pitch, velocity=velocity, time=0))
            
            track.append(Message('note_off', note=pitch, velocity=velocity, time=current_time))

            current_time = 0  # Zeit wird jetzt in delta-steps gezählt

  

    # Datei speichern
    mid.save(output_path)
    print(f"MIDI Saved to: {output_path}")
