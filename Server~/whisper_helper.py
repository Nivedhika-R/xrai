import time
import asyncio
from queue import Queue
from datetime import datetime, timedelta, timezone

import whisper
import torch
import soundfile as sf
import numpy as np

from threading import Thread
from aiortc import MediaStreamTrack
from av import AudioResampler

from logger import logger

import functools
whisper.torch.load = functools.partial(whisper.torch.load, weights_only=True)

logger.info("Loading Whisper model...")
model_name = "medium"
audio_model = None#whisper.load_model(model_name + ".en")
logger.info("Whisper model ready!")

class WhisperProcessor:
    def __init__(self, record_timeout=5, phrase_timeout=3):
        self.audio_model = audio_model
        self.data_queue = Queue()
        self.transcription = [""]
        self.phrase_time = None
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout

    def feed_audio(self, pcm_bytes: bytes):
        self.data_queue.put(pcm_bytes)

    def run(self):
        while True:
            now = datetime.now(timezone.utc)

            if not self.data_queue.empty():
                phrase_complete = False

                if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                    phrase_complete = True

                self.phrase_time = now

                # Combine all audio in the queue
                audio_data = b''.join(self.data_queue.queue)
                self.data_queue.queue.clear()

                # Convert to float32 PCM [-1.0, 1.0]
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe with Whisper
                result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en")
                text = result['text'].strip()

                if phrase_complete:
                    self.transcription.append(text)
                else:
                    self.transcription[-1] = text

                if text:
                    logger.info(f"\nUser said:\n{text}\n")

            time.sleep(0.1)

class RemoteAudioToWhisper(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.buffer = bytearray()
        self.stopped = False
        self.debug = False

        self.whisper_processor = WhisperProcessor()
        # self.whisper_thread = Thread(target=self.whisper_processor.run, daemon=True)
        # self.whisper_thread.start()

        self.sample_rate = 16000 # Whisper has a sample rate of 16000
        self.resampler = AudioResampler(
            format='s16',
            layout='mono',
            rate=self.sample_rate
        )

        if self.debug:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_file = f"debug_audio_{now}.wav"
            self.wav_writer = sf.SoundFile(
                self.debug_file, mode='w', samplerate=self.sample_rate, channels=1, subtype='PCM_16'
            )

    async def recv(self):
        if self.stopped:
            raise asyncio.CancelledError("Audio track stopped.")

        try:
            frame = await self.track.recv()
        except Exception as e:
            logger.warning(f"recv() failed or stopped: {e}")
            raise asyncio.CancelledError()

        if self.stopped:
            raise asyncio.CancelledError("Audio track stopped.")

        resampled_frames = self.resampler.resample(frame)
        if not isinstance(resampled_frames, list):
            resampled_frames = [resampled_frames]

        for resampled in resampled_frames:
            pcm_array = resampled.to_ndarray().astype(np.int16).flatten()
            pcm_bytes = pcm_array.tobytes()

            # write to WAV (optional for debugging)
            if self.debug:
                self.wav_writer.write(pcm_array)

            self.whisper_processor.feed_audio(pcm_bytes)

        return frame

    def stop(self):
        if self.stopped:
            return
        self.stopped = True

        if self.debug:
            self.wav_writer.close()

        super().stop()
