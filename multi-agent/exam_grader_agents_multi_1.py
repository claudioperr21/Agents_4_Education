from typing import List, Dict, Any
import os
import time
import json
import pdfplumber
import librosa
import pandas as pd
from pathlib import Path
from openai import OpenAIError, RateLimitError, APIConnectionError, Timeout
from dotenv import load_dotenv
import openai
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
import asyncio

# Load environment variables from .env (if present)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0
RUBRIC_CACHE_DIR = os.path.join(os.getcwd(), "cache")
os.makedirs(RUBRIC_CACHE_DIR, exist_ok=True)

# ========== AGENTS ==========
vc_pitch_agent = Agent(
    name="VC Pitch Agent",
    instructions=(
        "You are a seasoned VC pitch grader. For a {{duration}}-minute audio pitch, give each dimension a score from 1 (poor) to 10 (excellent), using the following anchors:\n"
        "\n1. Problem Clarity\n"
        "   • 1–3: No clear problem stated, listener confused\n"
        "   • 4–6: Problem mentioned but lacks context or urgency\n"
        "   • 7–8: Problem clearly described with context\n"
        "   • 9–10: Problem statement is crisp, impactful, and immediately compelling\n"
        "\n2. Market Evidence\n"
        "   • 1–3: No market data or vague claims\n"
        "   • 4–6: Qualitative market description, no numbers\n"
        "   • 7–8: One clear quantitative metric (TAM, growth rate)\n"
        "   • 9–10: Multiple strong data points (TAM, traction, growth) cited\n"
        "\n3. Solution Differentiation\n"
        "   • 1–3: Solution not differentiated, generic\n"
        "   • 4–6: Mentions a unique feature but no defense\n"
        "   • 7–8: Clearly highlights one defensible advantage\n"
        "   • 9–10: Demonstrates multiple, well-justified differentiators or proprietary edge\n"
        "\n4. Delivery & Pacing\n"
        "   • 1–3: Monotone or too fast/slow (outside 80–200 WPM), frequent long pauses (>30 %)\n"
        "   • 4–6: Understandable but some pacing issues (WPM 90–210, pauses 20–30 %)\n"
        "   • 7–8: Good pace (110–160 WPM), pauses <20 %\n"
        "   • 9–10: Engaging tone, ideal pacing (120–150 WPM), minimal pauses (<10 %)\n"
        "\nIf a rubric is provided, override this rubric with the one provided.\n"
        "\nReturn valid JSON EXACTLY in this format (no extra keys):\n"
        "{\n"
        "  \"Problem\": <1–10>,\n"
        "  \"Market\": <1–10>,\n"
        "  \"Solution\": <1–10>,\n"
        "  \"Delivery\": <1–10>,\n"
        "  \"Feedback\": \"<one sentence actionable feedback for each anchor>\",\n"
        "  \"total_score\": <1–10>,\n"
        "  \"overall_feedback\": \"<one paragraph summary of performance>\",\n"
        "  \"overall_justification\": \"<one paragraph justification for the overall grade>\"\n"
        "}\n"
        "\n### Rubric (if available):\n{rubric_markdown}"
    ),
    model=OpenAIChatCompletionsModel(
        model="gpt-4",
        openai_client=AsyncOpenAI()
    )
)

technical_exam_agent = Agent(
    name="Technical Exam Agent",
    instructions=(
        "You are a grading assistant for technical exams. Evaluate student responses based on the provided questions and, if available, the rubric.\n"
        "- For each question:\n"
        "  - Read the question and the student's answer.\n"
        "  - If a rubric is provided, follow it carefully to assign points.\n"
        "  - If no rubric is provided, use your expert-level knowledge to assess factual accuracy, completeness, and clarity.\n"
        "- Assign a score for each answer using a 0-10 point scale.\n"
        "- Provide feedback for each answer, including rationale and suggestions for improvement if needed.\n"
        "\n### Output Format:\nRespond ONLY with raw JSON (no markdown):\n"
        "{\n"
        "  \"question_1\": {\n    \"score\": X,\n    \"feedback\": \"...\"\n  },\n"
        "  \"question_2\": {\n    \"score\": Y,\n    \"feedback\": \"...\"\n  },\n"
        "  ...\n"
        "  \"total_score\": Z,\n"
        "  \"overall_feedback\": \"<one paragraph summary of performance>\",\n"
        "  \"overall_justification\": \"<one paragraph justification for the overall grade>\"\n"
        "}\n"
        "\n### Rubric (if available):\n{rubric_markdown}"
    ),
    model=OpenAIChatCompletionsModel(
        model="gpt-4",
        openai_client=AsyncOpenAI()
    )
)

narrative_exam_agent = Agent(
    name="Narrative Exam Agent",
    instructions=(
        "You are an exam grader. If a rubric is provided, use it to assign each question a numeric score (0-10) and give concise feedback.\n"
        "If no rubric is available, use your own criteria. Then compute the overall score as the average and provide general feedback.\n"
        "\nReturn JSON.\n"
        "\n### Output Format:\nRespond ONLY with raw JSON (no markdown):\n"
        "{\n"
        "  \"question_1\": {\n    \"score\": X,\n    \"feedback\": \"...\"\n  },\n"
        "  \"question_2\": {\n    \"score\": Y,\n    \"feedback\": \"...\"\n  },\n"
        "  ...\n"
        "  \"total_score\": Z,\n"
        "  \"overall_feedback\": \"<one paragraph summary of performance>\",\n"
        "  \"overall_justification\": \"<one paragraph justification for the overall grade>\"\n"
        "}\n"
        "\n### Rubric (if available):\n{rubric_markdown}"
    ),
    model=OpenAIChatCompletionsModel(
        model="gpt-4",
        openai_client=AsyncOpenAI()
    )
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "Route the input to the appropriate agent:\n"
        "- If it's an audio file (e.g., mp3), send to the VC Pitch Agent.\n"
        "- If it's a technical exam, send to the Technical Exam Agent.\n"
        "- If it's a narrative-style written exam, send to the Narrative Exam Agent."
    ),
    handoffs=[vc_pitch_agent, technical_exam_agent, narrative_exam_agent],
    model="gpt-4"
)

async def grade_input(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Convert dictionary to a string that the Triage Agent can understand
        exam_type = input_payload.get("exam_type")
        if exam_type == "vc_pitch":
            input_str = (
                f"Transcript: {input_payload['transcript']}\n"
                f"Duration: {input_payload['duration']:.2f} seconds\n"
                f"WPM: {input_payload['wpm']:.2f}\n"
                f"Silence Ratio: {input_payload['silence_ratio']:.2%}"
            )
        else:
            input_str = (
                f"Exam Type: {exam_type}\n"
                f"Questions:\n{input_payload['questions']}\n\n"
                f"Rubric:\n{input_payload['rubric']}\n\n"
                f"Student Responses:\n{input_payload['responses']}"
            )

        result = await Runner.run(triage_agent, input=input_str)
        return result.final_output
    except Exception as e:
        return {"error": str(e)}

# ========== UTILITIES ==========
def extract_pdf_to_markdown(pdf_path: str) -> str:
    def clean_text_formatting(text: str) -> str:
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            s = line.strip()
            if not s:
                cleaned.append("")
            elif s[0] in ("*", "•", "·", "-"):
                cleaned.append("- " + s.lstrip("*•·-").strip())
            else:
                cleaned.append(s)
        return "\n".join(cleaned) + "\n"

    def convert_table_to_markdown(table: List[List[str]]) -> str:
        header, *rows = table
        md = "| " + " | ".join(header) + " |\n"
        md += "| " + " | ".join("--" for _ in header) + " |\n"
        for r in rows:
            md += "| " + " | ".join(cell or "" for cell in r) + " |\n"
        return md

    out = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            out += f"\n\n## Page {i}\n"
            text = page.extract_text() or ""
            out += clean_text_formatting(text)
            for tbl in page.extract_tables() or []:
                out += "\n" + convert_table_to_markdown(tbl)
    return out


def transcribe(mp3_path: str) -> str:
    """Return transcript from Whisper with simple caching logic."""
    cache_file = os.path.join(RUBRIC_CACHE_DIR, os.path.basename(mp3_path) + ".txt")
    if os.path.exists(cache_file):
        return open(cache_file, "r").read()

    with open(mp3_path, "rb") as f:
        resp = openai.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="text"
        )
    transcript = resp if isinstance(resp, str) else resp.get("text", "")
    with open(cache_file, "w") as f:
        f.write(transcript)
    return transcript


def analyze_audio(mp3_path: str) -> Dict[str, Any]:
    y, sr = librosa.load(mp3_path, sr=16000, mono=True)
    duration = len(y) / sr
    transcript = transcribe(mp3_path)
    word_count = len(transcript.split())
    wpm = word_count / (duration / 60) if duration else 0
    intervals = librosa.effects.split(y, top_db=30)
    voiced = sum((e - s) for s, e in intervals) / sr
    silence_ratio = (duration - voiced) / duration if duration else 0
    return {
        "duration": duration,
        "wpm": wpm,
        "silence_ratio": silence_ratio,
        "transcript": transcript
    }

# ========== GRADERS ==========
def call_with_backoff(**kwargs):
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return openai.chat.completions.create(**kwargs)
        except (RateLimitError, OpenAIError, APIConnectionError, Timeout):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))

def grade_exam(rubric: str, questions: str, responses: str, exam_type: str = "narrative") -> dict:
    rubric_markdown = rubric.strip() or "No rubric provided."

    if exam_type == "technical":
        user_prompt = TECHNICAL_PROMPT_TEMPLATE.format(rubric_markdown=rubric_markdown) + f"\n\nQuestions:\n{questions}\n\nStudent Responses:\n{responses}"
        system_prompt = "You are a helpful technical exam grader."

    else:  # narrative default
        if rubric.strip():
            system_prompt = NARRATIVE_PROMPT_WITH_RUBRIC
            user_prompt = f"Rubric:\n{rubric}\n\nQuestions:\n{questions}\n\nStudent Responses:\n{responses}"
        else:
            system_prompt = NARRATIVE_PROMPT_NO_RUBRIC
            user_prompt = f"Questions:\n{questions}\n\nStudent Responses:\n{responses}"

    resp = call_with_backoff(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        seed=42
    )

    raw_response = resp.choices[0].message.content
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw": raw_response}