"""Shared WAV discovery for the Encode preview and preprocessing pipeline."""
import os


def find_wav_files(audio_dir):
    """Return sorted WAV paths recursively, without following directory symlinks."""
    paths = []
    for directory, _, names in os.walk(audio_dir, followlinks=False):
        for name in names:
            path = os.path.join(directory, name)
            if name.lower().endswith('.wav') and os.path.isfile(path):
                paths.append(path)
    return sorted(paths)
