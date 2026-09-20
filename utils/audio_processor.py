import yt_dlp
import os
import subprocess

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok = True)

""" Downloading the YouTube Video using its URL """
def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192"
            }
        ],
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        filename = os.path.splitext(filename)[0] + ".wav"

    return filename

""" COnvert any audio/video file to WAV format using FFMPEG """
def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    subprocess.run([
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-y",
        output_path
    ],
    check = True)

    return output_path

""" Converting the auio file into chunks of 10 mins """
def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    chunk_seconds = chunk_minutes * 60
    output_pattern = os.path.splitext(wav_path)[0] + "_chunk_%03d.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-i", wav_path,
            "-f", "segment",
            "-segment_time", str(chunk_seconds),
            "-c", "copy",
            output_pattern
        ],
        check=True
    )

    directory = os.path.dirname(wav_path) or "."
    filename = os.path.basename(os.path.splitext(wav_path)[0])

    chunks = [
        os.path.join(directory, file)
        for file in sorted(os.listdir(directory))
        if file.startswith(filename + "_chunk_")
        and file.endswith(".wav")
    ]

    return chunks

""" Final Processing of Input """
def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)

    else:
        print("Detected local file. COnverting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking Audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")

    return chunks