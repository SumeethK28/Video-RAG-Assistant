import streamlit as st
import tempfile
import os
import time

from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize_transcript, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="VideoMind AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Hero */

    .hero-title {
        font-size: 2.8rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }

    /* Cards */

    .dashboard-card {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 15px;
        background: rgba(128, 128, 128, 0.04);
    }

    .card-title {
        font-size: 0.85rem;
        color: #8b949e;
        margin-bottom: 7px;
    }

    .card-value {
        font-size: 1.5rem;
        font-weight: 650;
    }

    /* Result header */

    .meeting-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .meeting-caption {
        color: #8b949e;
        margin-bottom: 20px;
    }

    /* Chat */

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 8px;
    }

    /* Buttons */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    .stDownloadButton > button {
        border-radius: 10px;
    }

    /* Tabs */

    button[data-baseweb="tab"] {
        font-size: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "processed": False,
    "result": None,
    "messages": [],
    "source_name": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def save_uploaded_file(uploaded_file):

    suffix = os.path.splitext(uploaded_file.name)[1]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(uploaded_file.getvalue())
    temp_file.close()

    return temp_file.name


def reset_application():

    st.session_state.processed = False
    st.session_state.result = None
    st.session_state.messages = []
    st.session_state.source_name = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🎙️ VideoMind AI")

    st.caption(
        "AI-powered video and meeting intelligence"
    )

    st.divider()

    st.subheader("Input")

    input_type = st.radio(
        "Source",
        [
            "YouTube",
            "Upload"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    source = None


    # -------------------------
    # YOUTUBE
    # -------------------------

    if input_type == "YouTube":

        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="Paste YouTube URL..."
        )

        if youtube_url:
            source = youtube_url
            st.session_state.source_name = "YouTube Video"


    # -------------------------
    # FILE UPLOAD
    # -------------------------

    else:

        uploaded_file = st.file_uploader(
            "Upload audio/video",
            type=[
                "mp3",
                "wav",
                "mp4",
                "m4a",
                "webm",
                "mov"
            ]
        )

        if uploaded_file:

            source = save_uploaded_file(
                uploaded_file
            )

            st.session_state.source_name = (
                uploaded_file.name
            )


    st.divider()

    st.subheader("Transcription")

    language_display = st.selectbox(
        "Language",
        [
            "English",
            "Hindi / Hinglish"
        ]
    )

    if language_display == "Hindi / Hinglish":
        language = "hinglish"
    else:
        language = "english"


    if language == "english":

        st.caption(
            "English audio → Local Whisper"
        )

    else:

        st.caption(
            "Hindi/Hinglish → Sarvam AI → English"
        )


    st.divider()


    process_button = st.button(
        "✨ Analyze Content",
        type="primary",
        use_container_width=True
    )


    if st.session_state.processed:

        if st.button(
            "↻ New Analysis",
            use_container_width=True
        ):
            reset_application()
            st.rerun()


    st.divider()

    st.caption(
        "Powered by Whisper • Sarvam AI • Gemini • LangChain • ChromaDB"
    )


# =========================================================
# PROCESS PIPELINE
# =========================================================

if process_button:

    if not source:

        st.sidebar.warning(
            "Add a YouTube URL or upload a file."
        )

    else:

        try:

            progress = st.progress(0)

            with st.status(
                "AI analysis in progress...",
                expanded=True
            ) as status:

                # ---------------------------------
                # AUDIO PROCESSING
                # ---------------------------------

                st.write(
                    "🎧 Preparing audio..."
                )

                chunks = process_input(source)

                progress.progress(10)

                st.write(
                    f"✓ Audio prepared — "
                    f"{len(chunks)} chunks"
                )


                # ---------------------------------
                # TRANSCRIPTION
                # ---------------------------------

                st.write(
                    "📝 Transcribing content..."
                )

                transcript = transcribe_all(
                    chunks,
                    language=language
                )

                progress.progress(35)

                st.write(
                    "✓ Transcript generated"
                )


                # ---------------------------------
                # TITLE
                # ---------------------------------

                st.write(
                    "🏷️ Understanding content..."
                )

                title = generate_title(
                    transcript
                )

                progress.progress(45)


                # ---------------------------------
                # SUMMARY
                # ---------------------------------

                st.write(
                    "📋 Generating intelligent summary..."
                )

                summary = summarize_transcript(
                    transcript
                )

                progress.progress(60)


                # ---------------------------------
                # ACTION ITEMS
                # ---------------------------------

                st.write(
                    "🎯 Detecting action items..."
                )

                action_items = extract_action_items(
                    transcript
                )

                progress.progress(72)


                # ---------------------------------
                # DECISIONS
                # ---------------------------------

                st.write(
                    "💡 Identifying key decisions..."
                )

                decisions = extract_key_decisions(
                    transcript
                )

                progress.progress(82)


                # ---------------------------------
                # QUESTIONS
                # ---------------------------------

                st.write(
                    "❓ Finding unresolved questions..."
                )

                questions = extract_questions(
                    transcript
                )

                progress.progress(90)


                # ---------------------------------
                # RAG
                # ---------------------------------

                st.write(
                    "🧠 Building searchable knowledge base..."
                )

                rag_chain = build_rag_chain(
                    transcript
                )

                progress.progress(100)


                # ---------------------------------
                # STORE RESULT
                # ---------------------------------

                st.session_state.result = {

                    "title": title,

                    "transcript": transcript,

                    "summary": summary,

                    "action_items": action_items,

                    "key_decisions": decisions,

                    "open_questions": questions,

                    "rag_chain": rag_chain,

                    "language": language_display,

                    "chunks": len(chunks)
                }

                st.session_state.processed = True

                st.session_state.messages = []

                status.update(
                    label="Analysis completed successfully",
                    state="complete",
                    expanded=False
                )

            time.sleep(0.3)

            st.rerun()


        except Exception as e:

            st.error(
                f"Analysis failed: {e}"
            )


# =========================================================
# LANDING PAGE
# =========================================================

if not st.session_state.processed:

    st.markdown(
        """
        <div class="hero-title">
            Turn conversations into intelligence.
        </div>

        <div class="hero-subtitle">
            Transcribe, summarize, extract insights and chat
            with any meeting, lecture or video using AI.
        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-value">📝 Transcribe</div>
                <br>
                Convert English, Hindi and Hinglish audio
                into searchable text.
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-value">✨ Understand</div>
                <br>
                Generate summaries, decisions, action items
                and unresolved questions.
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-value">💬 Ask</div>
                <br>
                Chat directly with your content using
                retrieval-augmented generation.
            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    st.subheader("How it works")

    c1, c2, c3, c4 = st.columns(4)

    c1.info(
        "①\n\nAdd video or audio"
    )

    c2.info(
        "②\n\nAI transcribes content"
    )

    c3.info(
        "③\n\nInsights are extracted"
    )

    c4.info(
        "④\n\nAsk questions with RAG"
    )


# =========================================================
# RESULT DASHBOARD
# =========================================================

else:

    result = st.session_state.result


    # =====================================================
    # HEADER
    # =====================================================

    st.markdown(
        f"""
        <div class="meeting-title">
            {result["title"]}
        </div>

        <div class="meeting-caption">
            AI Content Intelligence Report
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # METRICS
    # =====================================================

    words = len(
        result["transcript"].split()
    )

    characters = len(
        result["transcript"]
    )


    metric1, metric2, metric3, metric4 = st.columns(4)


    metric1.metric(
        "Words",
        f"{words:,}"
    )

    metric2.metric(
        "Characters",
        f"{characters:,}"
    )

    metric3.metric(
        "Audio Chunks",
        result["chunks"]
    )

    metric4.metric(
        "Language",
        result["language"]
    )


    st.divider()


    # =====================================================
    # MAIN NAVIGATION
    # =====================================================

    (
        overview_tab,
        actions_tab,
        decisions_tab,
        questions_tab,
        transcript_tab,
        chat_tab
    ) = st.tabs(
        [
            "✨ Overview",
            "🎯 Action Items",
            "💡 Decisions",
            "❓ Questions",
            "📝 Transcript",
            "💬 AI Chat"
        ]
    )


    # =====================================================
    # OVERVIEW
    # =====================================================

    with overview_tab:

        left, right = st.columns(
            [2, 1]
        )


        with left:

            st.subheader(
                "Executive Summary"
            )

            st.markdown(
                result["summary"]
            )


        with right:

            st.subheader(
                "Content Details"
            )

            st.markdown(
                f"""
                **Source**

                {st.session_state.source_name}

                **Language**

                {result["language"]}

                **Transcript words**

                {words:,}

                **Processing chunks**

                {result["chunks"]}
                """
            )


        st.divider()


        st.subheader(
            "Quick Insights"
        )


        insight1, insight2, insight3 = st.columns(3)


        with insight1:

            with st.container(border=True):

                st.markdown(
                    "#### 🎯 Action Items"
                )

                st.markdown(
                    result["action_items"]
                )


        with insight2:

            with st.container(border=True):

                st.markdown(
                    "#### 💡 Key Decisions"
                )

                st.markdown(
                    result["key_decisions"]
                )


        with insight3:

            with st.container(border=True):

                st.markdown(
                    "#### ❓ Open Questions"
                )

                st.markdown(
                    result["open_questions"]
                )


    # =====================================================
    # ACTION ITEMS
    # =====================================================

    with actions_tab:

        st.subheader(
            "🎯 Action Items"
        )

        st.caption(
            "Tasks and responsibilities detected by AI."
        )

        with st.container(border=True):

            st.markdown(
                result["action_items"]
            )


    # =====================================================
    # DECISIONS
    # =====================================================

    with decisions_tab:

        st.subheader(
            "💡 Key Decisions"
        )

        st.caption(
            "Important decisions identified from the content."
        )

        with st.container(border=True):

            st.markdown(
                result["key_decisions"]
            )


    # =====================================================
    # QUESTIONS
    # =====================================================

    with questions_tab:

        st.subheader(
            "❓ Open Questions"
        )

        st.caption(
            "Questions or topics requiring follow-up."
        )

        with st.container(border=True):

            st.markdown(
                result["open_questions"]
            )


    # =====================================================
    # TRANSCRIPT
    # =====================================================

    with transcript_tab:

        top_left, top_right = st.columns(
            [4, 1]
        )


        with top_left:

            st.subheader(
                "📝 Full Transcript"
            )


        with top_right:

            st.download_button(
                "⬇ Download",
                data=result["transcript"],
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True
            )


        search_term = st.text_input(
            "Search transcript",
            placeholder="Search for a word or topic..."
        )


        if search_term:

            occurrences = (
                result["transcript"]
                .lower()
                .count(search_term.lower())
            )

            st.caption(
                f"{occurrences} occurrence(s) found"
            )


        st.text_area(
            "Transcript",
            value=result["transcript"],
            height=550,
            label_visibility="collapsed"
        )


    # =====================================================
    # AI CHAT
    # =====================================================

    with chat_tab:

        st.subheader(
            "💬 Chat with your content"
        )

        st.caption(
            "Answers are generated using the transcript as context."
        )


        # ---------------------------------------------
        # Empty Chat Suggestions
        # ---------------------------------------------

        if not st.session_state.messages:

            st.markdown(
                "**Try asking:**"
            )

            suggestion1, suggestion2, suggestion3 = st.columns(3)


            with suggestion1:

                if st.button(
                    "What are the main topics?",
                    use_container_width=True
                ):

                    st.session_state.pending_question = (
                        "What are the main topics discussed?"
                    )


            with suggestion2:

                if st.button(
                    "Explain the key concepts",
                    use_container_width=True
                ):

                    st.session_state.pending_question = (
                        "Explain the key concepts discussed."
                    )


            with suggestion3:

                if st.button(
                    "What should I remember?",
                    use_container_width=True
                ):

                    st.session_state.pending_question = (
                        "What are the most important things "
                        "I should remember?"
                    )


        # ---------------------------------------------
        # DISPLAY CHAT HISTORY
        # ---------------------------------------------

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )


        # ---------------------------------------------
        # CHAT INPUT
        # ---------------------------------------------

        question = st.chat_input(
            "Ask anything about this content..."
        )


        if "pending_question" in st.session_state:

            question = (
                st.session_state.pending_question
            )

            del st.session_state.pending_question


        if question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )


            with st.chat_message("user"):

                st.markdown(question)


            with st.chat_message("assistant"):

                with st.spinner(
                    "Thinking..."
                ):

                    try:

                        answer = ask_question(
                            result["rag_chain"],
                            question
                        )

                        st.markdown(answer)

                    except Exception as e:

                        answer = (
                            f"Unable to answer: {e}"
                        )

                        st.error(answer)


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )