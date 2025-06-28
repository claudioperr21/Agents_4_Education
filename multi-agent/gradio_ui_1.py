import gradio as gr
import os
import json
import asyncio
from exam_grader_agents_multi_1 import extract_pdf_to_markdown, analyze_audio, grade_input
from dotenv import load_dotenv
from fpdf import FPDF
import openai
from agents import set_tracing_export_api_key

load_dotenv()
set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))

# Utility function to extract text from various file formats
def extract_text_from_file(file_obj):
    if file_obj.name.endswith(".pdf"):
        return extract_pdf_to_markdown(file_obj.name)
    elif file_obj.name.endswith(".txt") or file_obj.name.endswith(".md"):
        return file_obj.read().decode("utf-8")
    else:
        return "Unsupported file format"

# Utility to export JSON result to PDF
def json_to_pdf(json_obj, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=10)
    for line in json.dumps(json_obj, indent=2, ensure_ascii=False).splitlines():
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)

# ========== GRADER HANDLERS ==========
def handle_exam(pdf_path, rubric_path, student_response_file, exam_type):
    questions_md = extract_pdf_to_markdown(pdf_path.name)
    rubric_md = extract_pdf_to_markdown(rubric_path.name) if rubric_path else ""
    student_response_md = extract_text_from_file(student_response_file)

    input_payload = {
        "questions": questions_md,
        "rubric": rubric_md,
        "responses": student_response_md,
        "exam_type": exam_type
    }

    result = asyncio.run(grade_input(input_payload))

    # Save results to JSON and PDF
    base_name = f"{exam_type}_grade_output"
    json_path = base_name + ".json"
    pdf_path = base_name + ".pdf"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    json_to_pdf(result, pdf_path)

    return json.dumps(result, indent=2, ensure_ascii=False), json_path, pdf_path

def handle_vc_pitch(audio_file):
    audio_metrics = analyze_audio(audio_file)
    input_payload = {
        "duration": audio_metrics["duration"],
        "wpm": audio_metrics["wpm"],
        "silence_ratio": audio_metrics["silence_ratio"],
        "transcript": audio_metrics["transcript"],
        "exam_type": "vc_pitch"
    }

    result = asyncio.run(grade_input(input_payload))

    base_name = "vc_pitch_grade_output"
    json_path = base_name + ".json"
    pdf_path = base_name + ".pdf"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    json_to_pdf(result, pdf_path)

    return json.dumps(result, indent=2, ensure_ascii=False), json_path, pdf_path

# ========== INTERFACES ==========
exam_tab = gr.Interface(
    fn=lambda pdf, rubric, student, exam_type: handle_exam(pdf, rubric, student, exam_type),
    inputs=[
        gr.File(label="Exam PDF"),
        gr.File(label="Rubric PDF (optional)"),
        gr.File(label="Student Response (.txt, .md, .pdf)"),
        gr.Radio(["narrative", "technical"], label="Exam Type")
    ],
    outputs=[
        gr.Textbox(label="Evaluation Output"),
        gr.File(label="Download JSON"),
        gr.File(label="Download PDF")
    ],
    title="Narrative & Technical Exam Grader"
)

vc_tab = gr.Interface(
    fn=handle_vc_pitch,
    inputs=[gr.Audio(label="Upload VC Pitch (MP3)", type="filepath")],
    outputs=[
        gr.Textbox(label="VC Pitch Evaluation Output"),
        gr.File(label="Download JSON"),
        gr.File(label="Download PDF")
    ],
    title="VC Pitch Grader"
)

gr.TabbedInterface([exam_tab, vc_tab], ["Text-based Exams", "VC Pitch Grading"]).launch()
