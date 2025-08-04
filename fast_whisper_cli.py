import argparse
import os
import time
import subprocess
import json
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed

import glob

def split_audio_ffmpeg(input_path, chunk_length_sec=3600, max_workers=8):
    base_name = os.path.splitext(input_path)[0]
    chunk_dir = f"{base_name}_chunk_{chunk_length_sec}"
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_pattern = os.path.join(chunk_dir, "chunk*.wav")
    existing_chunks = sorted(glob.glob(chunk_pattern))

    def get_duration(path):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())

    duration = get_duration(input_path)
    num_chunks = int(duration // chunk_length_sec) + 1

    if len(existing_chunks) == num_chunks:
        print(f"Using cached chunks in '{chunk_dir}': {len(existing_chunks)} files found.")
        return existing_chunks

    # Remove any partial chunks
    for f in existing_chunks:
        os.remove(f)

    def create_chunk(i):
        start_time = i * chunk_length_sec
        out_file = os.path.join(chunk_dir, f"chunk{i}.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-t", str(chunk_length_sec),
            "-i", input_path,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            out_file
        ]
        print(f"Creating chunk {i+1}/{num_chunks}: {out_file}")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_file

    chunks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_chunk, i) for i in range(num_chunks)]
        for future in as_completed(futures):
            chunks.append(future.result())

    # Sort chunks by filename to keep order
    chunks.sort()
    return chunks

def format_timestamp(seconds: float, fmt: str) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    if fmt == "srt":
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
    else:  # vtt
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

def main():
    parser = argparse.ArgumentParser(description="Whisper-like CLI using faster-whisper with chunked audio splitting via ffmpeg.")
    parser.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument("--model", default="small", help="Model size to use")
    parser.add_argument("--language", default=None, help="Language spoken in the audio")
    parser.add_argument("--output_dir", default="/output", help="Directory to save output files")
    parser.add_argument("--output_format", default="txt", choices=["txt", "srt", "vtt", "json", "log"], help="Output file format")
    parser.add_argument("--compute_type", default="int8", choices=["float32", "float16", "int8", "int8_float16"], help="Precision of the model")
    parser.add_argument("--beam_size", type=int, default=8, help="Beam size for transcription search")
    parser.add_argument("--chunk_length", type=int, default=1800, help="Chunk length in seconds (default 1800 = 30 minutes)")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model '{args.model}' with compute_type '{args.compute_type}' ...")
    model = WhisperModel(args.model, device, compute_type=args.compute_type)
    print("Model loaded.")

    os.makedirs(args.output_dir, exist_ok=True)

    for audio_path in args.audio:
        print(f"\nProcessing audio file: {audio_path}")

        # Split audio using ffmpeg without loading into memory
        chunks = split_audio_ffmpeg(audio_path, chunk_length_sec=args.chunk_length)
        print(f"Split into {len(chunks)} chunks.")

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}.{args.output_format}")

        if args.output_format == "json":
            all_segments = []
        else:
            f = open(output_path, "w", encoding="utf-8")

        segment_index = 1
        chunk_offset = 0.0  # seconds to add for timestamp adjustment

        for chunk_file in chunks:
            print(f"Transcribing chunk: {chunk_file}")
            start_chunk_time = time.time()

            segments, info = model.transcribe(chunk_file, language=args.language, beam_size=args.beam_size)
            for segment in segments:
                # <generator object WhisperModel.generate_segments at 0x...>
                # generate has no start and end attributes, so we need to adjust timestamps manually

                start_adj = segment.start + chunk_offset
                end_adj = segment.end + chunk_offset

                if args.output_format == "txt":
                    f.write(segment.text.strip() + "\n")
                    f.flush()
                elif args.output_format in ["srt", "vtt"]:
                    start_ts = format_timestamp(start_adj, args.output_format)
                    end_ts = format_timestamp(end_adj, args.output_format)
                    if args.output_format == "srt":
                        f.write(f"{segment_index}\n{start_ts} --> {end_ts}\n{segment.text.strip()}\n\n")
                    else:
                        f.write(f"{segment_index}\n{start_ts} --> {end_ts}\n{segment.text.strip()}\n\n")
                    f.flush()
                    segment_index += 1
                elif args.output_format == "json":
                    all_segments.append({
                        "start": start_adj,
                        "end": end_adj,
                        "text": segment.text
                    })
                else:
                    print(f"[{start_adj:.2f} - {end_adj:.2f}] {segment.text.strip()}")

            chunk_duration = args.chunk_length
            chunk_offset += chunk_duration

            end_chunk_time = time.time()
            print(f"Chunk {chunk_file} transcribed in {end_chunk_time - start_chunk_time:.2f} seconds.")

            # Cleanup chunk file to save disk space
            # os.remove(chunk_file)

        if args.output_format != "json":
            f.close()
        else:
            with open(output_path, "w", encoding="utf-8") as f_json:
                json.dump(all_segments, f_json, ensure_ascii=False, indent=2)

        # Optionally clean up chunks here
        for chunk_file in chunks:
            os.remove(chunk_file)
        print(f"Saved transcription to {output_path}")

if __name__ == "__main__":
    total_start = time.time()
    main()
    total_end = time.time()
    print(f"\nTotal execution time: {total_end - total_start:.2f} seconds.")