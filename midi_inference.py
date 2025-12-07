import os
os.system("cls")
print("importing.libs...")


import sys
import time
import argparse
import json
import torch
from mido import MidiFile, MidiTrack, Message
from helper.midi_transformer import GPTLike
import random
from helper.tokens_to_midi import tokens_to_midi

from datetime import datetime
from random import randrange

#-------------------------------

def ask_yes_no(prompt="Further? (y/n): "):
    yes = {"j", "ja", "y", "yes"}
    no = {"n", "nein", "no"}

    while True:
        answer = input(prompt).strip().lower()
        if answer in yes:
            return True
        if answer in no:
            return False
        print("Please only 'y' or 'n' entering.")


# -------------------------------
# Helper: Token -> Event Mapping
# -------------------------------
def load_token_map(vocab_json_path):
    with open(vocab_json_path, "r") as f:
        vocab_dict = json.load(f)
    token_map = {v: k for k, v in vocab_dict.items()}  # int -> event string
    return token_map



# -------------------------------
# Sampling Funktion
# -------------------------------

def progress_bar(current, total, bar_length=40):
    percent = current / total
    filled = int(bar_length * percent)
    bar = "#" * filled + "+" * (bar_length - filled)
    sys.stdout.write(f"   \rInference:  [{bar}] {percent*100:5.1f}%")
    sys.stdout.flush()


@torch.no_grad()

def sample_sequence(model, start_ids, length, reverse_vocab,
                    temperature=0.85, top_k=None,
                    device="cpu", max_context=1024,
                    max_polyphony=1,
                    min_time_interval=1,
                    min_pitch=60, max_pitch=84):

    model.eval()
    tokens = start_ids.copy()

    for i in range(length):

        # --- Fortschrittsbalken ---
        progress_bar(i + 1, length)

        # --- Kontext vorbereiten ---
        context = tokens[-max_context:]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

        logits = model(x)[:, -1, :] / temperature  # (1, vocab_size)

        # --- optional: Top-k filtering ---
        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, ix, v)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        token_str = reverse_vocab[next_id]

        # --- Pitch erkennen ---
        if token_str.startswith("NOTE_ON_"):
            part = token_str.split("_")[2]
            if part.isdigit():
                pitch = int(part)
                if pitch < min_pitch or pitch > max_pitch:
                    continue

        tokens.append(next_id)

    print()  # Zeilenumbruch nach Progressbar
    return tokens



def ok_sample_sequence(model, start_ids, length, reverse_vocab,
                    temperature=0.75, top_k=50, device="cuda",
                    max_context=1024, max_polyphony=1,min_time_interval=1,
                    min_pitch=60, max_pitch=84):
    model.eval()
    generated = start_ids.copy()
    vocab = {v: k for k, v in reverse_vocab.items()}
    active_notes = {}  # pitch -> token_id

    for _ in range(length):
        context = generated[-max_context:]
        context_tensor = torch.tensor([context], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(context_tensor)
            logits = logits[0, -1, :] / temperature

            # Top-K Sampling
            if top_k is not None:
                top_vals, top_idx = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits)
                probs[top_idx] = torch.softmax(top_vals, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1).item()
            token_str = reverse_vocab[next_id]

            print("Token: ", token_str," ", str(next_id))

            # NOTE_<pitch> Token clipping
            if token_str.startswith("NOTE_") and token_str[5:].isdigit():
                pitch = int(token_str.split("_")[1])
                         
                token_str = f"NOTE_{pitch}"
                next_id = vocab[token_str]
                active_notes[pitch] = next_id
                generated.append(next_id)
 

            # NOTE_OFF Token korrekt zuordnen
            if token_str == "NOTE_OFF":
                if active_notes:
                    # Beende die älteste aktive Note
                    pitch, token_on_id = active_notes.popitem()
                   
                    token_str = f"NOTE_OFF_{pitch}"
                    next_id = vocab.get(token_str, next_id)
                    #active_notes[pitch] = next_id
                    generated.append(next_id)  

                else:
                    # Kein aktiver Note-On, skip
                    continue
            

            # Doppelte TIME Tokens verhindern
            if token_str.startswith("TIME_") and len(generated) > 0 and generated[-1] == next_id:
                continue

            generated.append(next_id)

    return generated

# -------------------------------
# Main
# -------------------------------
def main():
    
    # -----main setup------------

    dir_path = os.path.dirname(os.path.realpath(__file__))

    checkpoint= dir_path + "/setup/" + "checkpoint_epoch50.pt"
    vocab_json = dir_path + "/setup/" + "vocabulary.json"
    
    length=1024
    device="cuda"
    min_note = 48
    max_note = 72
    max_polyphony=1
    min_time_interval=1
    bpm=130

    index=0
   
    start_ids=[]
    while True:

        
 
        os.system("cls")
        print("")

        index=index+1
        print("Midi Song Inference Iteration Nr.",str(index))
        print("")
 
        
        # random values loop
    
        #for a in range(1,4):
        #    start_ids.append(randrange(min_note,max_note))
        
        print("Do you want to input start melody?")

        # Beispiel:
        if ask_yes_no("Please enter Yes or No (y/n)? "):
            print("OK, we go on.")
            yes_flag=1        

        else:
            print("Canceled -  use old last Melody")
            yes_flag=0 
            #start_ids=[60,62,64,65,67,67] 
        
        if yes_flag==1:

            start_ids=[] 

            while yes_flag==1:
                note=input("Midi Note (0-127) or Enter: ")
                if note=="" or note==None:
                    print("Accepted.continue...") 
                    yes_flag=0
                    break 

                if int(note)<0:
                    note=0
                if int(note)>127:
                    note=127


                start_ids.append(int(note))
                print("Appended Midi Note: ",str(note))



        temperature = randrange(1,10)
        temperature = float(temperature/10)
        #temperature=0.85
        
        top_k = randrange(10,100)
        #top_k=80
        
        note_length = randrange(100,300)
        #note_length = 100
        
        print("Setup is: ",str(start_ids)," ",str(temperature)," ",str(top_k)," ",str(note_length))
        print("")
        a=input("wait for key to start...")


        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M%S")
        output=dir_path +"/" + "song_" + dt_string +".mid"


        # Vocabulary laden
        reverse_vocab = load_token_map(vocab_json)

        # Modell laden
        vocab_size = len(reverse_vocab)
        model = GPTLike(vocab_size=vocab_size, d_model=512, n_layers=6, n_heads=8, dropout=0.1, max_len=length) # 1024
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        # Sampling
        start_ids = start_ids if start_ids else [0]
    
        generated_tokens = sample_sequence(model, start_ids, length=500, reverse_vocab=reverse_vocab, temperature=temperature, top_k=top_k, device=device, max_context=1024, max_polyphony=max_polyphony, min_time_interval=min_time_interval)
    
        # Tokens -> MIDI
        tokens_to_midi(generated_tokens, reverse_vocab, output_path=output,min_pitch=min_note,max_pitch=max_note, bpm=bpm, note_length=note_length)

        # wait for user        
        a=input("wait Return key")


if __name__ == "__main__":
    main()
