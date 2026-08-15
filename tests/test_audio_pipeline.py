from pathlib import Path

import process_audio
from pydub import AudioSegment


class FakeAudio:
    channels = 1
    frame_rate = 16000

    def __len__(self):
        return 2000


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
