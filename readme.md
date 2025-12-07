
# MIDI Transformer Inference Tool

This project provides a Python-based MIDI generation tool using a GPT-like Transformer model.  
It loads a trained checkpoint, performs autoregressive token sampling, and converts the generated tokens into a playable MIDI file.

The script allows:
- Interactive input of a starting melody
- Randomized sampling parameters (temperature, top-k, note length)
- Real-time progress display
- Automatic MIDI file creation with timestamped filenames

---

## 🚀 Features

- GPT-style Transformer (`GPTLike`) for MIDI event generation  
- Token-to-event vocabulary system
- Interactive command-line UI
- Configurable sampling (temperature, top-k, context size)
- MIDI export using `mido`
- Pitch constraints (min/max note filtering)
- Optional manual seed melody input

---

## 📂 Project Structure

```

/
├── midi_inference.py        # This script (your provided code)
├── helper/
│   ├── midi_transformer.py
│   └── tokens_to_midi.py
└── setup/
├── checkpoint_epoch50.pt
└── vocabulary.json

````

---

## 🔧 Installation

### 1. Create a virtual environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main script:

```bash
python midi_inference.py
```

You will be prompted:

* Whether to input a start melody
* Optionally enter MIDI note numbers
* Sampling settings are partially randomized
* A generated MIDI file such as `song_25022025_154422.mid` will appear in the root directory

---

## 🎶 Output

Generated MIDI files are saved as:

```
song_<DDMMYYYY>_<HHMMSS>.mid
```

These files can be opened in any DAW or MIDI editor.

---

## 🧠 Model

The script loads:

* A vocabulary (`vocabulary.json`)
* A trained GPT-like transformer checkpoint (`checkpoint_epoch50.pt`)

Model parameters include:

* `d_model = 512`
* `n_layers = 6`
* `n_heads = 8`
* `dropout = 0.1`
* `max_len = 1024`

---

## ⚠️ Notes

* Requires CUDA if `device="cuda"` is used (modify to `"cpu"` if needed).
* Ensure `checkpoint_epoch50.pt` and `vocabulary.json` are present in the `setup/` directory.
* The script expects the `helper/` folder with the required modules.

---

## 📄 License

MIT License – feel free to use, modify, and share.

````

---

# 📄 **requirements.txt**

Based on imported modules, external libraries, and typical helper dependencies:

```text
torch==2.0.1
mido
python-rtmidi     # optional but often required by mido for playback
numpy
argparse
````

(Modules like `os`, `json`, `datetime`, `random`, `time`, `sys` are built-in.)

If `helper/midi_transformer.py` uses additional dependencies, add them as needed.

---

If you'd like, I can also generate:

✅ A GitHub-style project structure
✅ Badges (PyPI, license, build status)
✅ A more detailed README with examples, screenshots, or diagrams
Just tell me!

