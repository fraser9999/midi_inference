import sys
import numpy as np
import pygame
import pygame.midi

# ------------------------------
# Einstellungen
# ------------------------------
WINDOW_W, WINDOW_H = 700, 200
FPS = 60
DEFAULT_VELOCITY = 100
SAMPLE_RATE = 44100
BEEP_VOLUME = 0.3

pygame.init()
pygame.midi.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)

screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Großes MIDI-Keyboard (QWERTZ)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)
small_font = pygame.font.SysFont(None, 22)


# ------------------------------
# Weißtasten (ganze Noten)
# ------------------------------
WHITE_KEYS = {
    pygame.K_y: 48, pygame.K_u: 50, pygame.K_i: 52, pygame.K_o: 53,
    pygame.K_p: 55, pygame.K_LEFTBRACKET: 57, pygame.K_RIGHTBRACKET: 59,

    pygame.K_q: 60, pygame.K_w: 62, pygame.K_e: 64, pygame.K_r: 65,
    pygame.K_t: 67, pygame.K_z: 69, pygame.K_u: 71,
    pygame.K_i: 72, pygame.K_o: 74,

    pygame.K_1: 72, pygame.K_2: 74, pygame.K_3: 76, pygame.K_4: 77,
    pygame.K_5: 79, pygame.K_6: 81, pygame.K_7: 83,
    pygame.K_8: 84, pygame.K_9: 86, pygame.K_0: 88
}



# ------------------------------
# Schwarztasten (Halbtöne)
# ------------------------------
BLACK_KEYS = {
    pygame.K_a: 49, pygame.K_s: 51, pygame.K_d: 54, pygame.K_f: 56,
    pygame.K_g: 58, pygame.K_h: 61, pygame.K_j: 63, pygame.K_k: 66,
    pygame.K_l: 68, pygame.K_SEMICOLON: 70, pygame.K_QUOTE: 73
}



KEY_TO_NOTE = {**WHITE_KEYS, **BLACK_KEYS}

#--------------------------------

def print_pressed_notes(pressed_dict):
    if not pressed_dict:
        print("Aktive Noten: ---")
        return

    notes = [str(v[0]) for v in pressed_dict.values()]  # note only
    print("Aktive Noten:", " | ".join(notes))



# ------------------------------
# Beep-Funktion (Stereo)
# ------------------------------
def midi_to_frequency(n):
    return 440.0 * (2 ** ((n - 69) / 12.0))

def generate_beep(freq, duration=0.3):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(freq * t * 2 * np.pi)

    # int16 Mono
    mono = np.int16(wave * 32767)
    # Stereo
    stereo = np.column_stack((mono, mono))

    sound = pygame.sndarray.make_sound(stereo)
    sound.set_volume(BEEP_VOLUME)
    return sound


# ------------------------------
# MIDI-Funktionen
# ------------------------------
midi_out = None
for i in range(pygame.midi.get_count()):
    intf, name, inp, outp, opened = pygame.midi.get_device_info(i)
    if outp:
        midi_out = pygame.midi.Output(i)
        break

def note_on(n):
    if midi_out:
        midi_out.note_on(n, DEFAULT_VELOCITY)
    freq = midi_to_frequency(n)
    snd = generate_beep(freq)
    snd.play(-1)
    return snd

def note_off(n, snd):
    if midi_out:
        midi_out.note_off(n, 0)
    if snd:
        snd.stop()


# ------------------------------
# Hauptloop
# ------------------------------
pressed = {}  # key → (note, sound)
current_note = None
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # NOTE ON
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if event.key in KEY_TO_NOTE and event.key not in pressed:
                note = KEY_TO_NOTE[event.key]
                snd = note_on(note)
                pressed[event.key] = (note, snd)
                current_note = note
                print_pressed_notes(pressed)


        # NOTE OFF
        elif event.type == pygame.KEYUP:
            if event.key in pressed:
                note, snd = pressed.pop(event.key)
                note_off(note, snd)
                current_note = next(iter(pressed.values()))[0] if pressed else None

    

    # Anzeige zeichnen
    screen.fill((30, 30, 30))
    if current_note is not None:
        surf = font.render(f"Note: {current_note}", True, (255, 255, 255))
    else:
        surf = font.render("Keine Note", True, (255, 255, 255))
    screen.blit(surf, (20, 60))

    pygame.display.flip()
    clock.tick(FPS)


# Aufräumen
for n, snd in pressed.values():
    snd.stop()
if midi_out:
    midi_out.close()
pygame.midi.quit()
pygame.quit()
sys.exit()
