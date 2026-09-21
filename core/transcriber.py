import whisper
import os
import requests
import subprocess


SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = "saaras:v3"

_model = None


# --------------------------------------------------
# WHISPER
# --------------------------------------------------

def load_model():
    global _model

    if _model is None:
        print("Loading Whisper Model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully!!")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["text"].strip()


# --------------------------------------------------
# SARVAM AI
# --------------------------------------------------

def send_to_sarvam(piece_path: str) -> str:

    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set.")

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    data = {
        "model": "saaras:v3",
        "mode": "translate",
        "language_code": "hi-IN",
        "input_audio_codec": "pcm_s16le"
    }

    with open(piece_path, "rb") as audio_file:

        files = {
            "file": (
                os.path.basename(piece_path),
                audio_file,
                "audio/wav"
            )
        }

        response = requests.post(
            SARVAM_STT_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )

    if not response.ok:
        raise RuntimeError(
            f"Sarvam API Error {response.status_code}: "
            f"{response.text}"
        )

    result = response.json()

    transcript = result.get("transcript", "")

    # Empty transcript simply means no speech was detected
    if not transcript or not transcript.strip():
        return ""

    return transcript.strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:

    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set.")

    directory = os.path.dirname(chunk_path) or "."

    filename = os.path.basename(
        os.path.splitext(chunk_path)[0]
    )

    piece_pattern = os.path.join(
        directory,
        filename + "_sv_%03d.wav"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-loglevel", "error",
            "-i", chunk_path,
            "-f", "segment",
            "-segment_time", str(SARVAM_PIECE_SECONDS),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            "-y",
            piece_pattern
        ],
        check=True
    )

    pieces = [
        os.path.join(directory, file)
        for file in sorted(os.listdir(directory))
        if file.startswith(filename + "_sv_")
        and file.endswith(".wav")
    ]

    if not pieces:
        raise RuntimeError(
            f"No audio pieces created for {chunk_path}"
        )

    transcripts = []

    for i, piece_path in enumerate(pieces):

        print(
            f"  Sarvam piece {i + 1}/{len(pieces)}..."
        )

        try:
            text = send_to_sarvam(piece_path)

            # Add only if speech was detected
            if text:
                transcripts.append(text)

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(transcripts)


# --------------------------------------------------
# TRANSCRIPTION ROUTER
# --------------------------------------------------

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


# --------------------------------------------------
# TRANSCRIBE ALL CHUNKS
# --------------------------------------------------

def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    transcripts = []

    if language.lower() == "hinglish":
        print("Using Sarvam AI for transcription.")
    else:
        print("Using Whisper for transcription.")

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk "
            f"{i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language
        )

        transcripts.append(text)

    full_transcript = " ".join(transcripts).strip()

    if not full_transcript:
        raise RuntimeError(
            "Transcription completed but no text was produced."
        )

    print("Transcription Completed!!")

    return full_transcript