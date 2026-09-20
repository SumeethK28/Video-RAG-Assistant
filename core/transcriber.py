import whisper
import os
import requests
import subprocess
import glob

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None

def load_model():
    global _model

    if _model is None:
        print(f"Loading Model ....")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully!!")

    return _model

def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()

    result = model.transcribe(chunk_path, task = "transcribe")

    return result['text']

def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio.
    Split the chunk into 25-second pieces using FFmpeg,
    send each piece to Sarvam, and join the transcripts.
    """

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in environment / .env"
        )

    base_path = os.path.splitext(chunk_path)[0]
    piece_pattern = f"{base_path}_sv_%03d.wav"

    # Split audio into 25-second pieces
    subprocess.run(
        [
            "ffmpeg",
            "-i", chunk_path,
            "-f", "segment",
            "-segment_time", str(SARVAM_PIECE_SECONDS),
            "-c", "copy",
            "-y",
            piece_pattern
        ],
        check=True
    )

    # Get all generated pieces in order
    pieces = sorted(
        glob.glob(f"{base_path}_sv_*.wav")
    )

    full_text = ""

    for i, piece_path in enumerate(pieces):
        try:
            print(f"  → Sarvam piece {i + 1}/{len(pieces)} ...")

            transcript = _send_to_sarvam(piece_path)

            full_text += transcript + " "

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()

def transcribe_chunk(chunk_path: str, language: str = 'english') -> str:
    """
    Route one chunk to Whisper or Sarvam depending on the language choice.
    - English -> Whisper (local model)
    - Hinglish -> Sarvam (translates to English while transcribing)
    """
    if language.lower() == 'hinglish':
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)

def transcribe_all(chunks: list, language: str = 'english') -> str:
    full_transcript = ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)} ...")
        text = transcribe_chunk(chunk, language=language)

        full_transcript += text + " "

    print("Transcription Completed!!")

    return full_transcript.strip()