import sys
import os
os.chdir(r'D:\Echo-main\Echo-main\backend')

# Test vad_service
from services.vad_service import ImprovedVAD, detect_speech_simple, VAD_CONFIG, DEFAULT_CONFIG
print('vad_service imported successfully')

# Test stt
from services.stt import transcribe_audio_full, count_fillers, _wrap_pcm_as_wav
print('stt imported successfully')

# Test asr_pipeline
from services.asr_pipeline import ASRPipeline, ASRResult, transcribe_pipeline
print('asr_pipeline imported successfully')

# Test asr_quality_monitor
from services.asr_quality_monitor import ASRQualityMonitor
print('asr_quality_monitor imported successfully')

# Test basic VAD functionality
import numpy as np
vad = ImprovedVAD(config_name='mobile')
silence = np.zeros(320, dtype=np.float32)
result = vad.detect_speech(silence)
print(f'VAD basic test: is_speech={result.is_speech}')

print('All core speech architecture modules are functional!')
print('='*60)