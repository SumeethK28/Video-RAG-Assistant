""" Actionable Items, Decisions, Questions """
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os

from core.summarize import split_transcript

def get_llm():
    return ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key = os.getenv("GOOGLE_API_KEY"), temperature = 0.2)

def build_chain(system_prompt: str):
    llm = get_llm()

    return (RunnablePassthrough() | RunnableLambda(lambda x: {"text" : x}) | ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ]) | llm | StrOutputParser())

def extract_action_items(transcript: str) -> str:
    llm = get_llm()

    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, extract all action items. For each provide: \n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a number list. If none found say 'No action items found.'"
    ) 

    chunks = split_transcript(transcript)

    chunk_actions = []

    for chunk in chunks:
        result = chain.invoke(chunk)
        chunk_actions.append(result)

    combined = "\n\n".join(chunk_actions)

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert meeting analyst. Combine these partial "
            "action items into one final list. Remove duplicate action items "
            "caused by overlapping transcript chunks. Do not add any new "
            "information. For each action item provide the task description, "
            "owner, and deadline. Format as a numbered list."),
        ("human", "{text}")
    ])

    combined_chain = (RunnablePassthrough() | RunnableLambda(lambda x: {"text" : x}) | combined_prompt | llm | StrOutputParser())

    return combined_chain.invoke(combined)

def extract_key_decisions(transcript: str) -> str:
    llm = get_llm()

    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, extract all key decisions made. "
        "Format as a numbered list. If none found say 'No key decisions found.'"
    )

    chunks = split_transcript(transcript)

    chunk_decisions = []

    for chunk in chunks:
        result = chain.invoke(chunk)
        chunk_decisions.append(result)

    combined = "\n\n".join(chunk_decisions)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting analyst. Combine these partial key decisions "
            "into one final list. Remove duplicate decisions caused by overlapping "
            "transcript chunks. Do not add any new information. "
            "Format as a numbered list. If none found say 'No key decisions found.'"
        ),
        ("human", "{text}")
    ])

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)

def extract_questions(transcript: str) -> str:
    llm = get_llm()
    
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions or topics "
        "needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )

    chunks = split_transcript(transcript)

    chunk_questions = []

    for chunk in chunks:
        result = chain.invoke(chunk)
        chunk_questions.append(result)

    combined = "\n\n".join(chunk_questions)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting analyst. Combine these partial open questions "
            "into one final list. Remove duplicate questions caused by overlapping "
            "transcript chunks. Do not add any new information. "
            "Format as a numbered list. If none found say 'No open questions found.'"
        ),
        ("human", "{text}")
    ])

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)