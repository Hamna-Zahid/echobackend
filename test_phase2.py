#!/usr/bin/env python3
import sys
sys.path.insert(0, r'D:\Echo-main\Echo-main\backend')

# Test all imports
from services.vad_service import ImprovedVAD, detect_speech_simple
from services.stt import transcribe_audio_full, count_fillers
from services.asr_pipeline import ASRPipeline, ASRResult, transcribe_pipeline
from services.asr_quality_monitor import ASRQualityMonitor
from services.domain_adaptation import CoachLanguageAdapter, get_coach_adapter, rerank_coach_transcript

print('OK: All Phase 2 modules import successfully')

# Test domain adaptation
adapter = get_coach_adapter()
result = rerank_coach_transcript('I think the session is going well', 75.0)
terms = result['terms_detected']
conf = result['confidence']
print(f'OK: Domain adaptation: detected terms={terms}, confidence={conf}')

# Test VAD
vad = ImprovedVAD(config_name='mobile')
import numpy as np
silence = np.zeros(320, dtype=np.float32)
vad_result = vad.detect_speech(silence)
print(f'OK: VAD: is_speech={vad_result.is_speech}')

print()
print('Phase 2 basic functionality verified!')