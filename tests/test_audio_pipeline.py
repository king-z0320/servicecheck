from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import process_audio
import pytest
from pydub import AudioSegment


class FakeAudio:
    channels = 1
    frame_rate = 16000

    def __len__(self):
        return 2000


def test_model_loaders_disable_remote_update_checks(monkeypatch):
    calls = []

    def auto_model(**kwargs):
        calls.append(kwargs)
        return object()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "funasr",
        SimpleNamespace(AutoModel=auto_model),
    )

    process_audio.load_asr_model()
    process_audio.load_emotion_model()

    assert len(calls) == 2
    assert all(call["disable_update"] is True for call in calls)


def test_convert_audio_uses_bundled_ffmpeg_without_system_ffprobe(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.m4a"
    source.touch()
    output = tmp_path / "converted.wav"
    calls = []

    monkeypatch.setattr(process_audio.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        "imageio_ffmpeg.get_ffmpeg_exe",
        lambda: "C:/bundled/ffmpeg.exe",
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output.touch()

    monkeypatch.setattr(process_audio.subprocess, "run", fake_run)
    monkeypatch.setattr(
        AudioSegment,
        "from_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pydub must not require ffprobe in bundled mode")
        ),
    )
    monkeypatch.setattr(AudioSegment, "from_wav", lambda *_: FakeAudio())

    audio, duration = process_audio.convert_m4a_to_wav(source, output)

    assert audio.channels == 1
    assert duration == 2.0
    assert calls[0][0] == [
        "C:/bundled/ffmpeg.exe",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output),
    ]
    assert calls[0][1] == {"check": True, "capture_output": True}


def test_process_single_audio_keeps_all_outputs_in_explicit_directory(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.m4a"
    source.touch()
    output_dir = tmp_path / "derived"
    output_dir.mkdir()
    generated_paths = []

    monkeypatch.setattr(
        process_audio,
        "convert_m4a_to_wav",
        lambda input_path, output_path: (FakeAudio(), 2.0),
    )
    monkeypatch.setattr(
        process_audio,
        "run_asr_with_speaker_diarization",
        lambda wav_path: [],
    )
    monkeypatch.setattr(
        process_audio,
        "parse_asr_result",
        lambda result: [
            {"speaker": "坐席", "text": "测试", "start": 0, "end": 1}
        ],
    )
    monkeypatch.setattr(process_audio, "run_emotion_recognition", lambda _: None)
    monkeypatch.setattr(
        process_audio,
        "generate_demo_data_js",
        lambda transcript, audio_info, emotion, output_path, base_name: (
            generated_paths.append(output_path) or {"transcript": transcript}
        ),
    )

    result = process_audio.process_single_audio(
        source,
        output_dir,
        demo_js_output_path=None,
    )

    assert generated_paths == [output_dir / "demo_data_source.js"]
    assert Path(result["json_transcript"]).parent == output_dir
    assert Path(result["wav"]).parent == output_dir


def test_long_emotion_audio_is_chunked_and_scores_are_duration_weighted(tmp_path):
    wav_path = tmp_path / "long.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(10)
        wav.writeframes(b"\x00\x00" * 65)

    calls = []

    class Model:
        def generate(self, *, input, granularity, extract_embedding):
            calls.append(Path(input))
            scores = ([1.0, 0.0], [0.0, 1.0], [1.0, 0.0])[len(calls) - 1]
            return [{"labels": ["neutral", "angry"], "scores": scores}]

    result = process_audio.run_emotion_with_model(
        wav_path,
        Model(),
        max_chunk_seconds=3,
        chunk_root=tmp_path,
    )

    assert len(calls) == 3
    assert result[0]["labels"] == ["neutral", "angry"]
    assert result[0]["scores"] == pytest.approx([3.5 / 6.5, 3.0 / 6.5])
    assert all(not path.exists() for path in calls)
