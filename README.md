# openwakeword-colab-2026-fixed

A Python-3.13/Colab-compatible fork of [alfiedennen/openwakeword-colab-2026](https://github.com/alfiedennen/openwakeword-colab-2026) — a bulletproof, run-all-and-walk-away notebook for training [openWakeWord](https://github.com/dscripka/openWakeWord) custom wake-word models.

I used this to train a custom wake word ("Orion") for a personal AI-assistant project. The original notebook fails out of the box on the current Colab runtime (Python 3.13) — this fork patches 4 additional breakages found while debugging a real training run.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Muzzammil-Ahmed-009/openwakeword-colab-2026/blob/main/train_wakeword.ipynb)

## Why this fork exists
The original repo's own README says it patches "at least eight" known Colab breakage modes as of 2026-05. By August 2026, Colab's default runtime had moved to **Python 3.13**, which broke it again in 4 new ways:

| # | Break | Symptom | Fix |
|---|---|---|---|
| 1 | `piper-phonemize-cross` has no Python 3.13 wheel | `ModuleNotFoundError: No module named 'piper_phonemize'` | Swapped to `piper-phonemize-fix` (community fork with cp313 wheels) |
| 2 | `speexdsp-ns` (openwakeword's optional dep) has no cp313 wheel | `pip install -e ./openwakeword` silently fails; surfaces later as `ModuleNotFoundError: No module named 'openwakeword'` inside the training subprocess | Install openwakeword with `--no-deps`, always re-run the install (not gated behind a "does the folder exist" check) |
| 3 | `deep-phonemizer` missing from install list entirely (also reported [upstream #1](https://github.com/alfiedennen/openwakeword-colab-2026/issues/1)) | `ModuleNotFoundError: No module named 'dp'` during negative clip generation | Added `deep-phonemizer==0.0.19` to the install cell |
| 4 | `deep-phonemizer`'s checkpoint loader wasn't updated for PyTorch 2.6+'s `weights_only=True` default | `_pickle.UnpicklingError: Weights only load failed` | Runtime-patch `dp/model/model.py` to pass `weights_only=False` (safe — checkpoint is from deep-phonemizer's own trusted source) |

Full writeup / debugging story: [LinkedIn post link — add once posted]

I've also opened an [upstream PR](https://github.com/alfiedennen/openwakeword-colab-2026/pull/3) with these same fixes.

## Quickstart
Same as upstream — click the Colab badge above, set Runtime → GPU (T4 free, or L4 for Colab Pro), edit `TARGET_PHRASE` / `MODEL_NAME` in the training-config cell, then Runtime → Run all.

## Credits
All underlying design/methodology credit goes to the original authors:
- [openwakeword-colab-2026](https://github.com/alfiedennen/openwakeword-colab-2026) by alfiedennen — the notebook this is forked from
- [openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) by Rhasspy
- ACAV100M, MIT environmental impulse responses, FMA — data sources (see upstream README for full list)

## License
MIT — inherited from the upstream repo. See `LICENSE`.
