"""Voice pipeline (Layer 2): wake word, speech-to-text, and text-to-speech.

Everything here imports heavier optional deps (sounddevice, numpy, openwakeword,
faster-whisper, edge-tts). Import these modules lazily/guarded so the app still
runs with only the Layer 1 dependencies installed.
"""
