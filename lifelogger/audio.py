"""
Audio capture → VAD → transcription pipeline.

Two audio streams:
  1. Microphone (your voice)
  2. System loopback (other people's voices, videos, meetings)

silero-vad gates recording — only transcribe when someone is actually talking.
faster-whisper does the transcription.

Each stream runs in its own thread. Completed speech segments are queued
for the transcription worker thread.
"""

import os
import queue
import struct
import threading
import time
import wave
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SAMPLES = 512  # silero-vad requires exactly 512 samples at 16kHz
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 16-bit = 2 bytes per sample

# Where to store audio clips
AUDIO_DIR = Path.home() / "lifelogger" / "audio"


class AudioPipeline:
    def __init__(self, db):
        self.db = db
        self._running = False
        self._transcribe_queue = queue.Queue()
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    def start(self):
        self._running = True

        # Load models once
        print("[audio] Loading VAD model...")
        self._load_vad()
        print("[audio] Loading Whisper model (this takes a moment)...")
        self._load_whisper()
        print("[audio] Models loaded.")

        # Start transcription worker
        threading.Thread(
            target=self._transcribe_worker, daemon=True, name="transcriber"
        ).start()

        # Start audio streams
        threading.Thread(
            target=self._capture_stream, args=("mic",),
            daemon=True, name="audio-mic"
        ).start()

        threading.Thread(
            target=self._capture_stream, args=("loopback",),
            daemon=True, name="audio-loopback"
        ).start()

        print("[audio] Capturing mic + system audio")

    def stop(self):
        self._running = False

    def _load_vad(self):
        from silero_vad import load_silero_vad, VADIterator
        self._vad_model = load_silero_vad()
        # Each stream gets its own VADIterator (stateful)
        self._vad_mic = VADIterator(
            self._vad_model, sampling_rate=SAMPLE_RATE, threshold=0.5
        )
        self._vad_loopback = VADIterator(
            self._vad_model, sampling_rate=SAMPLE_RATE, threshold=0.5
        )

    def _load_whisper(self):
        from faster_whisper import WhisperModel
        # Use "base" for fast startup, "medium" for better accuracy
        # Change to "medium" once you confirm it works
        model_size = os.environ.get("WHISPER_MODEL", "base")
        self._whisper = WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )

    def _get_pyaudio(self):
        """Import PyAudioWPatch (falls back to regular PyAudio)."""
        try:
            import pyaudiowpatch as pyaudio
            return pyaudio
        except ImportError:
            import pyaudio
            return pyaudio

    def _open_mic_stream(self, pa):
        """Open the default microphone input."""
        return pa.open(
            format=pa.get_format_from_width(2),
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SAMPLES,
        )

    def _open_loopback_stream(self, pa):
        """Open the system audio loopback device (Windows WASAPI)."""
        pyaudio_mod = self._get_pyaudio()

        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio_mod.paWASAPI)
        except OSError:
            print("[audio] WASAPI not available — loopback disabled")
            return None

        default_output = pa.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        # Find the loopback virtual device for the default speakers
        loopback_dev = None
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if (dev["name"].startswith(default_output["name"])
                    and "Loopback" in dev["name"]):
                loopback_dev = dev
                break

        if not loopback_dev:
            print("[audio] No loopback device found — loopback disabled")
            return None

        # Loopback device may have different native rate — we resample later
        native_rate = int(loopback_dev["defaultSampleRate"])
        native_channels = max(1, int(loopback_dev["maxInputChannels"]))

        stream = pa.open(
            format=pa.get_format_from_width(2),
            channels=native_channels,
            rate=native_rate,
            input=True,
            input_device_index=int(loopback_dev["index"]),
            frames_per_buffer=CHUNK_SAMPLES,
        )

        # Store native params for resampling
        stream._native_rate = native_rate
        stream._native_channels = native_channels
        return stream

    def _capture_stream(self, source):
        """Capture loop for one audio stream (mic or loopback)."""
        pyaudio_mod = self._get_pyaudio()
        pa = pyaudio_mod.PyAudio()

        try:
            if source == "mic":
                stream = self._open_mic_stream(pa)
                vad = self._vad_mic
            else:
                stream = self._open_loopback_stream(pa)
                vad = self._vad_loopback
                if stream is None:
                    return
        except Exception as e:
            print(f"[audio] Failed to open {source} stream: {e}")
            return

        native_rate = getattr(stream, "_native_rate", SAMPLE_RATE)
        native_channels = getattr(stream, "_native_channels", CHANNELS)

        speech_buffer = []
        speech_start = None
        in_speech = False

        while self._running:
            try:
                raw = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            except Exception as e:
                print(f"[audio] {source} read error: {e}")
                time.sleep(0.1)
                continue

            # Convert to numpy, resample if needed
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # Downmix to mono if stereo
            if native_channels > 1:
                audio = audio.reshape(-1, native_channels).mean(axis=1)

            # Resample to 16kHz if needed
            if native_rate != SAMPLE_RATE:
                ratio = SAMPLE_RATE / native_rate
                new_len = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_len)
                audio = np.interp(indices, np.arange(len(audio)), audio)

            # Feed chunks of exactly 512 samples to VAD
            import torch
            chunk_tensor = torch.from_numpy(audio[:CHUNK_SAMPLES].copy())
            if len(chunk_tensor) < CHUNK_SAMPLES:
                continue

            event = vad(chunk_tensor, return_seconds=True)

            if event:
                if "start" in event:
                    in_speech = True
                    speech_start = datetime.now().isoformat()
                    speech_buffer = [audio]
                elif "end" in event:
                    if in_speech:
                        speech_buffer.append(audio)
                        speech_end = datetime.now().isoformat()
                        # Combine buffer and queue for transcription
                        full_audio = np.concatenate(speech_buffer)
                        self._transcribe_queue.put(
                            (source, speech_start, speech_end, full_audio)
                        )
                    in_speech = False
                    speech_buffer = []
            elif in_speech:
                speech_buffer.append(audio)

                # Safety: don't buffer more than 5 minutes of audio
                max_chunks = (5 * 60 * SAMPLE_RATE) // CHUNK_SAMPLES
                if len(speech_buffer) > max_chunks:
                    speech_end = datetime.now().isoformat()
                    full_audio = np.concatenate(speech_buffer)
                    self._transcribe_queue.put(
                        (source, speech_start, speech_end, full_audio)
                    )
                    speech_start = datetime.now().isoformat()
                    speech_buffer = []

        stream.stop_stream()
        stream.close()
        pa.terminate()

    def _transcribe_worker(self):
        """Drain the transcription queue — run whisper on each segment."""
        while self._running:
            try:
                source, ts_start, ts_end, audio = self._transcribe_queue.get(timeout=2)
            except queue.Empty:
                continue

            # Save audio clip
            filename = f"{source}_{ts_start.replace(':', '-')}.wav"
            audio_path = str(AUDIO_DIR / filename)
            try:
                self._save_wav(audio_path, audio)
            except Exception as e:
                print(f"[audio] Failed to save {audio_path}: {e}")
                audio_path = None

            # Transcribe
            try:
                segments, info = self._whisper.transcribe(
                    audio, beam_size=5, language="en"
                )
                text = " ".join(seg.text.strip() for seg in segments)
            except Exception as e:
                print(f"[audio] Transcription error: {e}")
                text = f"[transcription failed: {e}]"

            if text.strip():
                self.db.log_transcript(ts_start, ts_end, source, text, audio_path)
                print(f"[audio] [{source}] {text[:80]}...")

    def _save_wav(self, path, audio_float32):
        """Save float32 numpy audio to a 16-bit WAV file."""
        audio_int16 = (audio_float32 * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
