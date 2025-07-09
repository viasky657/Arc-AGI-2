"""
Real-time Voice Streaming Module for EnhancedCTMDiffusion

This module integrates real-time audio streaming capabilities with the
EnhancedCTMDiffusion model, leveraging its dynamic binary patch entropy method
for efficient processing of voice data.
"""

import torch
import numpy as np
import pyaudio

from .ctm_Diffusion_NEWNEW import EnhancedCTMDiffusion, DynamicEntropyPatcher, EnhancedCTMConfig

class RealtimeVoiceStreamer:
    def __init__(self, model: EnhancedCTMDiffusion, config: EnhancedCTMConfig):
        self.model = model
        self.config = config
        self.patcher = DynamicEntropyPatcher(
            embedding_dim=config.patch_embedding_dim,
            patch_cnn_channels=config.patch_encoder_cnn_channels,
            patching_mode=config.entropy_patcher_threshold_type,
            global_threshold=config.entropy_patcher_global_threshold,
            relative_threshold=config.entropy_patcher_relative_threshold,
            min_patch_size=config.entropy_patcher_min_patch_size,
            max_patch_size=config.entropy_patcher_max_patch_size,
        )
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start_stream(self, chunk_size=1024, sample_rate=16000):
        """Starts the audio stream for real-time input."""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()

    def stop_stream(self):
        """Stops the audio stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback to process incoming audio data."""
        self.process_audio_chunk(in_data)
        return (in_data, pyaudio.paContinue)

    def process_audio_chunk(self, audio_chunk):
        """Processes a chunk of audio data."""
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)

        # Use dynamic entropy patcher to create patches
        patches, _, _ = self.patcher(audio_tensor)

        # Get generated output from the model
        with torch.no_grad():
            generated_output = self.model.iterative_ctm_diffusion_sample(
                shape=patches.shape,
                initial_byte_sequence_for_inference=patches
            )
        
        # In a real application, you would play this output back
        print("Generated output shape:", generated_output[0].shape)

    def run(self, duration=10, chunk_size=1024, sample_rate=16000):
        """Runs the real-time voice streaming loop."""
        self.start_stream(chunk_size, sample_rate)
        print("Streaming for", duration, "seconds...")
        pyaudio.time.sleep(duration)
        self.stop_stream()
        print("Streaming finished.")

if __name__ == '__main__':
    # Example usage
    config = EnhancedCTMConfig()
    model = EnhancedCTMDiffusion(config)
    
    streamer = RealtimeVoiceStreamer(model, config)
    streamer.run()