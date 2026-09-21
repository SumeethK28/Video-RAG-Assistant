# VideoMind AI

VideoMind AI turns a YouTube video or an uploaded audio/video file into a searchable intelligence report. It combines speech recognition, language-model analysis, local embeddings, and retrieval-augmented generation so that long meetings, lectures, and tutorials can be understood and queried from one place.

The project has two interfaces:

- A Streamlit dashboard for interactive analysis and chat.
- A command-line workflow in `main.py` for scripted processing and terminal-based questions.

## What It Does

### 1. Accepts video and audio

The application can process:

- YouTube URLs.
- Uploaded files in the Streamlit app: `mp3`, `wav`, `mp4`, `m4a`, `webm`, and `mov`.
- Local audio/video paths through the command-line workflow.

YouTube audio is downloaded with `yt-dlp`. Local files are converted with FFmpeg. Audio is normalized to mono, 16 kHz, signed 16-bit PCM WAV and split into ten-minute chunks for reliable processing.

### 2. Transcribes English, Hindi, and Hinglish

The transcription engine uses two routes:

- **English:** OpenAI Whisper runs locally. The model defaults to `small` and can be changed with `WHISPER_MODEL`.
- **Hindi/Hinglish:** Sarvam AI processes 25-second audio pieces and translates the result into English. The Sarvam route uses the `saaras:v3` model and Hindi (`hi-IN`) as its input language.

Temporary Sarvam pieces are removed after each request. The final transcript is rejected if every transcription request produces empty text.

### 3. Generates structured insights

Google Gemini, accessed through LangChain, produces:

- A short title.
- A professional bullet-point summary.
- Action items with task, owner, and deadline when available.
- Key decisions.
- Unresolved questions and follow-up topics.

Long transcripts are split into overlapping sections before analysis. Partial results are combined to reduce context-size pressure and remove duplicate findings caused by overlap.

### 4. Supports transcript-grounded questions

The RAG workflow:

1. Splits the transcript into 500-character chunks with 50-character overlap.
2. Creates document objects with chunk metadata.
3. Embeds the chunks using `all-MiniLM-L6-v2` on the CPU.
4. Stores them in a persistent Chroma collection named `meeting_transcript`.
5. Retrieves the four most similar chunks for each question.
6. Sends only the retrieved transcript context to Gemini.

The assistant is instructed not to invent information. When an answer is not present in the retrieved transcript context, it responds that the information could not be found.

## Streamlit Dashboard

Start the interactive application from the project root:

```powershell
streamlit run app.py
```

The dashboard provides:

- YouTube URL input or local file upload.
- English or Hindi/Hinglish language selection.
- Progress updates for downloading, chunking, transcription, analysis, and indexing.
- Overview with title, summary, content statistics, and quick insights.
- Separate tabs for action items, decisions, questions, and the full transcript.
- Transcript search with occurrence counts.
- Transcript download as `transcript.txt`.
- Suggested questions and free-form AI chat grounded in the processed content.
- A reset action for starting a new analysis.

## Command-Line Workflow

Run the interactive terminal workflow with:

```powershell
python main.py
```

It asks for a YouTube URL or local file path and then asks for either `english` or `hinglish`. After processing, it prints the title, summary, extracted insights, and opens a terminal chat loop. Type `exit`, `quit`, or `q` to leave the chat.

## Installation

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The project also requires these system-level tools:

- **FFmpeg**, available on `PATH`, for audio conversion and segmentation.
- **Node.js**, available on `PATH`, for the JavaScript runtime used by `yt-dlp` during YouTube extraction.

Whisper downloads its selected model the first time it is used. The Hugging Face embedding model may also be downloaded on first use.

## Environment Variables

Create a `.env` file in the project root. Do not commit it.

```env
GOOGLE_API_KEY=your_google_generative_ai_key
SARVAM_API_KEY=your_sarvam_api_key
WHISPER_MODEL=small
```

`GOOGLE_API_KEY` is required for titles, summaries, extraction, and RAG answers. `SARVAM_API_KEY` is required only for Hindi/Hinglish transcription. `WHISPER_MODEL` is optional; larger Whisper models can improve accuracy but require more memory and processing time.

## Project Layout

```text
.
├── app.py                    # Streamlit user interface
├── main.py                   # Command-line pipeline and RAG chat
├── requirements.txt          # Python dependencies
├── core/
│   ├── extractor.py          # Action items, decisions, and questions
│   ├── rag_engine.py         # Retrieval-augmented question answering
│   ├── summarize.py          # Transcript splitting, title, and summary
│   ├── transcriber.py        # Whisper/Sarvam transcription routing
│   └── vector_store.py       # Chroma and Hugging Face embeddings
├── utils/
│   └── audio_processor.py    # Download, conversion, and audio chunking
├── downloads/                # Generated media and temporary audio pieces
└── vector_db/                # Persistent Chroma data
```

`downloads/` and `vector_db/` are ignored by Git because they contain generated media, temporary files, embeddings, and local database data.

## End-to-End Data Flow

```text
YouTube URL or uploaded file
	↓
yt-dlp / FFmpeg normalization
	↓
10-minute WAV chunks
	↓
Whisper or Sarvam transcription
	↓
Transcript
   ┌────┼───────────────┬───────────────┐
   ↓    ↓               ↓               ↓
Title  Summary    Structured insights  Chroma index
					   ↓
				      RAG chat
```

## Troubleshooting

### `No transcript was produced`

Check that:

1. `SARVAM_API_KEY` exists in `.env` for Hindi/Hinglish mode.
2. FFmpeg is installed and available from the terminal.
3. The generated audio is not silent or corrupted.
4. The Sarvam response is successful and contains a non-empty `transcript` field.

Sarvam input is deliberately converted to mono, 16 kHz PCM WAV and divided into pieces of no more than 25 seconds because the synchronous API has a short audio-duration limit.

### YouTube reports that no JavaScript runtime is available

Install Node.js and ensure this command works:

```powershell
node --version
```

The downloader is configured to use Node. Restart the terminal after installing Node if the command is not recognized.

### Gemini reports `contents are required`

This means an empty transcript reached the language model. Fix the transcription/API problem first; title generation now rejects empty input instead of sending an invalid Gemini request.

### Existing vector data is stale

The Chroma database is persisted in `vector_db/`. To rebuild it for a new transcript, stop the application and remove that directory before running a new analysis. It is generated data and is intentionally excluded from version control.

## Technology Stack

- Python
- Streamlit
- LangChain LCEL
- Google Gemini through `langchain-google-genai`
- OpenAI Whisper
- Sarvam AI Speech-to-Text
- FFmpeg and `yt-dlp`
- ChromaDB
- Hugging Face `all-MiniLM-L6-v2` embeddings
- PyTorch

## Current Scope

The application is designed for one active analysis at a time in the Streamlit session. Its vector store is local and persisted on disk, while Gemini and Sarvam requests require network access and valid API credentials. The system is transcript-based: it analyzes spoken audio and does not currently perform visual frame or slide understanding.
