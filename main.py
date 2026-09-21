from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize_transcript, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Video Assistant...")

    chunks = process_input(source)

    transcript = transcribe_all(chunks, language=language)
    print("Raw Transcription [First 300 Characters]", transcript[:300])

    title = generate_title(transcript)

    summary = summarize_transcript(transcript)

    action_items = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)

    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain
    }

if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"TITLE: {result['title']}")
    print(f"\n SUMMARY: \n{result['summary']}")
    print(f"\n ACTION ITEMS: \n{result['action_items']}")
    print(f"\n KEY DECISIONS: \n{result['key_decisions']}")
    print(f"\n OPEN QUESTIONS: \n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 - Chat with your meeting via RAG
    print("\n Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result['rag_chain']

    while True:
        question = input("You: ").strip()
        if question.lower() in ['exit', 'quit', 'q']:
            print("Visit Again!!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n Assistant: {answer}\n")