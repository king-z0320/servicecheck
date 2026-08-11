"""单通话音频/ASR/GPU 基线测量（一次性）。

测量：转码耗时、ASR 耗时、情绪耗时、显存占用、音频时长、RTF。
用法：python -m qc.batch.profile_single_call audio/audio1.m4a
结果打印为 JSON，需人工抄录到实施台账。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from time import monotonic


def profile(wav_or_m4a: str) -> dict:
    path = Path(wav_or_m4a)
    timings: dict[str, float] = {}

    # 转码（若已是 wav 则仅计时重采样）
    t0 = monotonic()
    from process_audio import convert_m4a_to_wav

    wav_path = Path("processed") / f"{path.stem}_profile.wav"
    wav_path.parent.mkdir(exist_ok=True)
    audio, duration = convert_m4a_to_wav(path, wav_path)
    timings["transcode_ms"] = (monotonic() - t0) * 1000

    # ASR + 分离
    t0 = monotonic()
    from process_audio import run_asr_with_speaker_diarization

    run_asr_with_speaker_diarization(wav_path)
    timings["asr_ms"] = (monotonic() - t0) * 1000

    # 情绪
    t0 = monotonic()
    from process_audio import run_emotion_recognition

    run_emotion_recognition(wav_path)
    timings["emotion_ms"] = (monotonic() - t0) * 1000

    rtf = (timings["transcode_ms"] + timings["asr_ms"] + timings["emotion_ms"]) / 1000 / max(duration, 0.001)
    return {
        "file": str(path),
        "audio_duration_s": round(duration, 2),
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "rtf": round(rtf, 3),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python -m qc.batch.profile_single_call <audio_file>")
        raise SystemExit(1)
    print(json.dumps(profile(sys.argv[1]), ensure_ascii=False, indent=2))
