# openwakeword-colab-2026

A bulletproof, run-all-and-walk-away Colab notebook for training [openWakeWord](https://github.com/dscripka/openWakeWord) models in 2026.

**Train your own custom wake word in ~75-90 minutes on Colab Pro. Three lines to edit, one button to press. Supports 36 languages via Piper TTS.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pafernanr/openwakeword-colab-2026/blob/main/train_wakeword.ipynb)

---

## Why this notebook exists

openWakeWord's official Colab notebook has bit-rotted hard since 2023. As of 2026, on a default Colab runtime, it fails out of the box for at least eight separate reasons — Python 3.12 has no `piper-phonemize` wheels, `torchaudio 2.x` removed `set_audio_backend`, the example YAML config keys silently moved to required-or-`KeyError`, the `piper-sample-generator` package layout changed, etc.

This notebook is patched against all of them. **Every cell is self-healing** — checks its own outputs, re-creates what's missing. There's a hard pre-flight gate (cell 4) that fails loudly before any of the slow downloads start, so you know within 60 seconds whether the run will succeed.

It also **replaces openwakeword's `auto_train` with a hand-rolled PyTorch loop** that mirrors `auto_train`'s curriculum exactly:

- 3-stage learning rate (1e-4 → 1e-5 → 1e-6)
- Negative-weight ramp 1 → 1500 over training
- Hard-negative mining (only train on `loss > 0.001` negatives + `loss < 0.999` positives)
- FP-per-hour validation against an ACAV100M continuous-audio slice (~11 hours of speech, music, noise)
- 90/90/10 percentile checkpoint ensemble averaging
- Sigmoid baked into the exported ONNX (so APK runtime threshold of 0.5 means what you think it means)

The `auto_train` upstream path keeps surfacing bugs against `mmap_batch_generator` shape handling on current openwakeword `master`. The hand-rolled trainer in cell 14 is small enough (~250 lines) to read in one sitting and debug if anything ever changes.

## Quickstart

1. Click the **Open in Colab** badge above (or upload `train_wakeword.ipynb` to your own Colab).
2. **Runtime → Change runtime type → L4 GPU + High RAM** (Colab Pro, $10/mo). Free T4 also works but is ~2× slower. A100 doesn't help — training is network/CPU-bound, not GPU-bound.
3. **Cell 0 — edit three lines:**
   ```python
   TARGET_PHRASE = ['mr graves', 'mister graves']   # what your wake word is
   MODEL_NAME    = 'mr_graves'                       # output filename + dirs
   LANGUAGE      = 'en_US'                           # any Piper locale (36 languages)
   ```
4. **Runtime → Run all**.
5. Walk away ~75-90 min. The last cell auto-downloads `<MODEL_NAME>.onnx` and `<MODEL_NAME>.tflite`.
6. Drop the model(s) into your APK / on-device app's wakeword assets dir.

## Validated example

I trained this exact pipeline on `'mr graves'` for [Harold](https://github.com/alfiedennen/harold-road), a personal home-assistant Pixel app. Real-world test on the device:

```
ONNX models loaded (wake = wakewords/mr_graves.onnx)
WAKE — score=0.99664426
WAKE — score=0.59243464
WAKE — score=0.65439160
WAKE — score=0.77693284
WAKE — score=0.85433600
WAKE — score=0.64701974
WAKE — score=0.80747485
WAKE — score=0.95243360
WAKE — score=0.99938180
WAKE — score=0.90104705
WAKE — score=0.99639570
WAKE — score=0.98879445
WAKE — score=0.51673140
```

13 utterances, 13 fires (across distance, volume, "Mr"/"Mister" variants). Scores 0.52–0.999, median ~0.85. **Zero phantom fires** during 30 minutes of normal phone use (TV, music, conversation, kitchen sounds). Matches openwakeword's pretrained Hey Jarvis baseline (0.984/0.989) on clear utterances.

The previous attempt — an over-simplified hand-rolled trainer that skipped most of `auto_train`'s curriculum — scored "0.9 acc / 0.87 recall on the test split" but **fired every 2-3 seconds on background noise in production**. The 4000-sample balanced test split lied; production exposed the model to ~1000× more diverse negatives than any test split contains. That's why the curriculum matters.

## What's in the notebook

| Cell | Est. time | What |
|---|---|---|
| **0. Config** | instant | **Edit `TARGET_PHRASE`, `MODEL_NAME`, and `LANGUAGE`** |
| **1. Install** | ~2 min | apt + pip in correct order; `piper-phonemize-cross` first, `piper-tts --no-deps` last, `piper-sample-generator` v3+ for multilingual |
| **2. Clone + voices + patches** | ~2 min | Clone openwakeword, download TTS voices (libritts .pt for en_US, auto-discovered .onnx for 35 other languages), apply 5 runtime patches |
| **3. Pre-flight** | instant | Hard-fails if any dep/file is missing — gate before slow downloads |
| **4. Download data** | ~15 min | Embedding models + MIT impulse responses + FMA music + ACAV features. All with resume |
| **5. Prepare audio** | ~6 min | FMA MP3 → 16 kHz WAV conversion + ACAV subsample (1.7 GB train + 170 MB val) |
| **6. Config + clips** | ~15 min | Build training YAML, patch deep-phonemizer, generate TTS clips, resample to 16 kHz |
| **7. Augment + train** | ~40 min | Augment with noise/reverb, extract features, 3-stage training with FP/hour validation |
| **8. Export + download** | ~1 min | Ensemble checkpoints, export ONNX + TFLite, verify, browser download |

## What can go wrong (and how to fix)

**Recall too low (<18/20 fires)?** Bump `n_samples` in cell 6 to 5000. Or set `target_recall` from 0.5 to 0.7. Or add `augmentation_rounds: 2` for more variety per positive clip.

**FP rate too high (>1 phantom per 30 min real-life)?** Bump `max_negative_weight` from 1500 → 3000. Or record 30 min of YOUR target ambient (TV, your speaker setup, your kitchen) and append features to `acav_val_subset.npy` so the trainer's FP/hour metric reflects what matters to you.

**Threshold tuning?** The exported ONNX has sigmoid baked in, so model outputs are in [0, 1] and 0.5 is a sane default. If your runtime supports it, raise to 0.6 or 0.7 for fewer false positives at the cost of some recall.

## Costs

- **Colab Pro**: $10/month — gets you L4 GPU + High RAM + ~1× compute units per training run
- **One training run**: ~75-90 minutes wallclock = roughly 1/100th of your monthly compute units. You can iterate dozens of times per month.
- **Free Colab T4** also works but takes ~2× longer (~2.5 hrs) and the runtime can disconnect if your tab is backgrounded.

## Caveats

- **Wake word phrases matter.** A 2-syllable phrase that sounds like nothing in English (`mr graves` works because "graves" is uncommon in everyday speech) generalizes way better than a common word ("hello", "play", "stop"). Pick something that's not in your normal vocabulary.
- **The model is binary.** All entries in `TARGET_PHRASE` activate the same single output. You can't train one model with multiple distinct wake words; you'd train multiple ONNX files and run them in parallel.
- **Multilingual but not polyglot.** Each training run targets one language. Set `LANGUAGE` to any Piper locale (`es_ES`, `de_DE`, `fr_FR`, etc.) and the notebook auto-downloads matching voices. But a model trained on Spanish clips won't fire on the same phrase spoken with an English accent — train one model per language you need.

## License

MIT. See `LICENSE`.

## Credits

- [openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka — the underlying training methodology + shared embedding models.
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) by Rhasspy — synthetic positive clip generation via Piper TTS.
- [ACAV100M](https://acav100m.github.io/) — large-scale negative audio corpus.
- [MIT environmental impulse responses](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses) — reverb augmentation.
- [FMA](https://github.com/mdeff/fma) — Free Music Archive small subset for background noise mixing.

This notebook represents about 6-8 hours of debugging across two failed attempts and one successful one. The lessons baked into the cells (every patch, every idempotency check, every gotcha comment) reflect specific bugs hit and resolved on a real Colab Pro runtime in 2026-05. If you hit a NEW failure mode, please open an issue.
