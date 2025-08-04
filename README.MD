```markdown
# Faster Whisper Docker CLI

This repository provides a Dockerized CLI tool for efficient speech transcription using [faster-whisper](https://github.com/guillaumekln/faster-whisper). It leverages NVIDIA CUDA for GPU acceleration and splits long audio files into chunks to enable scalable transcription with `ffmpeg`.

---

## Features

- Uses NVIDIA CUDA 12.6 with cuDNN and Python 3.11 for GPU-accelerated transcription.
- Supports chunked audio splitting with `ffmpeg` to handle long audio files without memory overload.
- Supports multiple output formats: `txt`, `srt`, `vtt`, `json`, and `log`.
- Configurable model size, language, beam size, chunk length, and compute precision.
- Automatically cleans up temporary audio chunks after transcription.
- Supports batch transcription of multiple audio files.

---

## Docker Image

This image is based on the official NVIDIA CUDA 12.6 runtime image with cuDNN on Ubuntu 22.04, with Python 3.11 and `faster-whisper` pre-installed.

---

## Usage

### Build the Docker Image

```bash
docker build -t faster-whisper-cli .
```

### Run Transcription

```bash
docker run --rm --gpus all -v /path/to/audio:/audio -v /path/to/output:/output faster-whisper-cli \
    /audio/your_audio_file.wav \
    --model small \
    --output_dir /output \
    --output_format srt \
    --language en \
    --chunk_length 1800 \
    --compute_type int8 \
    --beam_size 8
```

---

## Arguments

- `audio` (positional): One or more audio file paths to transcribe.
- `--model`: Model size to use (default: `small`). Options depend on `faster-whisper` models available.
- `--language`: Language spoken in the audio (default: autodetect).
- `--output_dir`: Directory to save transcription files (default: `/output`).
- `--output_format`: Output file format (`txt`, `srt`, `vtt`, `json`, `log`). Default: `txt`.
- `--compute_type`: Model precision (`float32`, `float16`, `int8`, `int8_float16`). Default: `int8`.
- `--beam_size`: Beam search size for transcription (default: 8).
- `--chunk_length`: Length of audio chunks in seconds (default: 1800 seconds = 30 minutes).

---

## How It Works

1. **Chunking:**  
   The input audio is split into chunks of specified length using `ffmpeg`. This helps handle long audio files efficiently.

2. **Transcription:**  
   Each chunk is transcribed independently using `faster-whisper` on GPU (or CPU fallback).

3. **Output Generation:**  
   Transcriptions from all chunks are combined and saved in the specified output format.

4. **Cleanup:**  
   Temporary chunk files are deleted after transcription to save disk space.

---

## Requirements

- NVIDIA GPU with CUDA support.
- Docker with NVIDIA Container Toolkit installed ([instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).
- Audio files accessible as Docker volumes.

---

## Notes

- The default chunk length is 30 minutes to balance memory usage and transcription speed.
- Output directory inside the container defaults to `/output`; ensure you map a host directory to it.
- GPU acceleration requires `--gpus all` flag in `docker run`.
- The container installs `torch`, `torchvision`, `torchaudio`, and `faster-whisper` Python packages within the image.

---

## Example

Transcribe a two-hour audio file in English into an SRT subtitle file:

```bash
docker run --rm --gpus all -v /home/user/audio:/audio -v /home/user/output:/output faster-whisper-cli \
    /audio/two_hour_audio.mp3 \
    --model medium \
    --language en \
    --output_dir /output \
    --output_format srt \
    --chunk_length 3600
```

---

## License

This project leverages open-source components; please refer to their respective licenses.

---

## Acknowledgements

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for the fast Whisper implementation.
- NVIDIA for CUDA Docker base images.

---

If you encounter issues or have suggestions, feel free to open an issue or pull request.
```