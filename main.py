import math
import os
from dotenv import load_dotenv
import re
import base64
import io
import contextlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from scipy.stats import norm, t as t_dist
import requests
import json
from datetime import date, datetime

load_dotenv()
# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Inferential Statistics Learning App",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGO_PATH = "nec_logo.png"
COUNTER_FILE = "visitor_count.json"
FEEDBACK_FILE = "feedback.json"


# ─────────────────────────────────────────────
# Visitor Counter
# ─────────────────────────────────────────────
def _load_counter() -> dict:
    if os.path.exists(COUNTER_FILE):
        try:
            with open(COUNTER_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"total": 0, "today": 0, "last_date": ""}


def _save_counter(data: dict):
    with open(COUNTER_FILE, "w") as f:
        json.dump(data, f)


def increment_visitor() -> dict:
    if "visited" not in st.session_state:
        st.session_state.visited = True
        data = _load_counter()
        today_str = str(date.today())
        data["total"] = data.get("total", 0) + 1
        if data.get("last_date") == today_str:
            data["today"] = data.get("today", 0) + 1
        else:
            data["today"] = 1
            data["last_date"] = today_str
        _save_counter(data)
    return _load_counter()


# ─────────────────────────────────────────────
# Feedback – Likes + Comments
# ─────────────────────────────────────────────
def _load_feedback() -> dict:
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"likes": 0, "comments": []}


def _save_feedback(data: dict):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_like():
    data = _load_feedback()
    data["likes"] = data.get("likes", 0) + 1
    _save_feedback(data)


def add_comment(name: str, comment: str, rating: int):
    data = _load_feedback()
    data.setdefault("comments", []).append({
        "name": name.strip() or "Anonymous",
        "comment": comment.strip(),
        "rating": rating,
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    })
    _save_feedback(data)


# ─────────────────────────────────────────────
# PDF Generation
# ─────────────────────────────────────────────
def generate_qa_pdf(topic: str, question: str, answer: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    base = getSampleStyleSheet()
    PURPLE = colors.HexColor("#7c3aed")
    INDIGO = colors.HexColor("#1e1b4b")
    GREEN = colors.HexColor("#065f46")
    LGRAY = colors.HexColor("#f8fafc")
    BORDER = colors.HexColor("#e2e8f0")

    title_style = ParagraphStyle("T2", parent=base["Title"],
                                 textColor=INDIGO, fontSize=20, spaceAfter=4, alignment=TA_CENTER)
    sub_style = ParagraphStyle("Sub", parent=base["Normal"],
                               textColor=PURPLE, fontSize=11, spaceAfter=2, alignment=TA_CENTER)
    label_style = ParagraphStyle("Lbl", parent=base["Normal"],
                                 textColor=PURPLE, fontSize=10, fontName="Helvetica-Bold", spaceAfter=4)
    body_style = ParagraphStyle("Bdy", parent=base["Normal"],
                                textColor=colors.HexColor("#1e293b"), fontSize=10, leading=16)
    ans_style = ParagraphStyle("Ans", parent=base["Normal"],
                               textColor=GREEN, fontSize=10, leading=16)
    meta_style = ParagraphStyle("Meta", parent=base["Normal"],
                                textColor=colors.HexColor("#94a3b8"), fontSize=8, alignment=TA_CENTER)

    def safe(t):
        return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("σ", "sigma").replace("μ", "mu").replace("α", "alpha")
                .replace("x̄", "x-bar").replace("√", "sqrt").replace("×", "x")
                .replace("±", "+/-").replace("≤", "<=").replace("≥", ">=")
                .replace("≠", "!=").replace("≈", "~=").replace("−", "-")
                .replace("•", "-").replace("→", "->").replace("₀", "0")
                .replace("₁", "1").replace("₂", "2"))

    story = []
    story.append(Paragraph("National Engineering College", title_style))
    story.append(Paragraph("Inferential Statistics Learning App", sub_style))
    story.append(Paragraph("https://nec.edu.in  |  AI&amp;DS Department", sub_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=PURPLE))
    story.append(Spacer(1, 0.4 * cm))

    # Topic badge
    tb_data = [[Paragraph(f"Topic:  {safe(topic)}",
                          ParagraphStyle("TB", parent=base["Normal"],
                                         textColor=colors.white, fontSize=11,
                                         fontName="Helvetica-Bold"))]]
    tb = Table(tb_data, colWidths=[doc.width])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PURPLE),
                            ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                            ("LEFTPADDING", (0, 0), (-1, -1), 14)]))
    story.append(tb);
    story.append(Spacer(1, 0.4 * cm))

    # Question
    story.append(Paragraph("Question", label_style))
    q_t = Table([[Paragraph(safe(question), body_style)]], colWidths=[doc.width])
    q_t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LGRAY), ("BOX", (0, 0), (-1, -1), 1, BORDER),
                             ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                             ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))
    story.append(q_t);
    story.append(Spacer(1, 0.35 * cm))

    # Answer
    story.append(Paragraph("AI Tutor Answer", label_style))
    answer_html = "<br/>".join(
        re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe(ln)) if ln.strip() else "&nbsp;"
        for ln in answer.split("\n")
    )
    a_t = Table([[Paragraph(answer_html, ans_style)]], colWidths=[doc.width])
    a_t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
                             ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#86efac")),
                             ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                             ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14)]))
    story.append(a_t);
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%d %B %Y at %I:%M %p')} | "
        "National Engineering College | AI&amp;DS Department", meta_style))
    doc.build(story)
    buf.seek(0)
    return buf.read()


def show_pdf_download(topic: str, question: str, answer: str, key_suffix: str = ""):
    try:
        pdf_bytes = generate_qa_pdf(topic, question, answer)
        st.download_button(
            label="📄 Download Q&A as PDF",
            data=pdf_bytes,
            file_name=f"NEC_Stats_{topic.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"pdf_{key_suffix}_{abs(hash(question)) % 99991}",
        )
    except Exception as e:
        st.warning(f"PDF error: {e}")


def generate_calculation_pdf(topic: str, inputs: dict, steps: list, results: dict, figure_bytes: bytes = None) -> bytes:
    """Generate PDF for manual calculation results with optional figure."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle, Image
    from reportlab.lib.enums import TA_CENTER

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    base = getSampleStyleSheet()
    PURPLE = colors.HexColor("#7c3aed")
    INDIGO = colors.HexColor("#1e1b4b")
    GREEN = colors.HexColor("#065f46")
    BLUE = colors.HexColor("#1d4ed8")
    LGRAY = colors.HexColor("#f8fafc")
    BORDER = colors.HexColor("#e2e8f0")

    title_style = ParagraphStyle("T2", parent=base["Title"],
                                 textColor=INDIGO, fontSize=20, spaceAfter=4, alignment=TA_CENTER)
    sub_style = ParagraphStyle("Sub", parent=base["Normal"],
                               textColor=PURPLE, fontSize=11, spaceAfter=2, alignment=TA_CENTER)
    label_style = ParagraphStyle("Lbl", parent=base["Normal"],
                                 textColor=PURPLE, fontSize=10, fontName="Helvetica-Bold", spaceAfter=4)
    body_style = ParagraphStyle("Bdy", parent=base["Normal"],
                                textColor=colors.HexColor("#1e293b"), fontSize=10, leading=16)
    step_style = ParagraphStyle("Step", parent=base["Normal"],
                                textColor=BLUE, fontSize=10, leading=16)
    result_style = ParagraphStyle("Res", parent=base["Normal"],
                                  textColor=GREEN, fontSize=11, fontName="Helvetica-Bold", leading=16)
    meta_style = ParagraphStyle("Meta", parent=base["Normal"],
                                textColor=colors.HexColor("#94a3b8"), fontSize=8, alignment=TA_CENTER)

    def safe(t):
        if t is None:
            return ""
        return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("σ", "sigma").replace("μ", "mu").replace("α", "alpha")
                .replace("x̄", "x-bar").replace("√", "sqrt").replace("×", "x")
                .replace("±", "+/-").replace("≤", "<=").replace("≥", ">=")
                .replace("≠", "!=").replace("≈", "~=").replace("−", "-")
                .replace("•", "-").replace("→", "->").replace("₀", "0")
                .replace("₁", "1").replace("₂", "2"))

    story = []
    story.append(Paragraph("National Engineering College", title_style))
    story.append(Paragraph("Inferential Statistics Learning App", sub_style))
    story.append(Paragraph("https://nec.edu.in  |  AI&amp;DS Department", sub_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=PURPLE))
    story.append(Spacer(1, 0.4 * cm))

    # Topic badge
    tb_data = [[Paragraph(f"Manual Calculation:  {safe(topic)}",
                          ParagraphStyle("TB", parent=base["Normal"],
                                         textColor=colors.white, fontSize=11,
                                         fontName="Helvetica-Bold"))]]
    tb = Table(tb_data, colWidths=[doc.width])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PURPLE),
                            ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                            ("LEFTPADDING", (0, 0), (-1, -1), 14)]))
    story.append(tb);
    story.append(Spacer(1, 0.4 * cm))

    # Inputs section
    if inputs:
        story.append(Paragraph("Input Values", label_style))
        input_data = [[Paragraph(f"<b>{safe(k)}:</b> {safe(v)}", body_style)] for k, v in inputs.items()]
        input_table = Table(input_data, colWidths=[doc.width])
        input_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LGRAY),
                                         ("BOX", (0, 0), (-1, -1), 1, BORDER),
                                         ("TOPPADDING", (0, 0), (-1, -1), 8),
                                         ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 12)]))
        story.append(input_table);
        story.append(Spacer(1, 0.35 * cm))

    # Steps section
    if steps:
        story.append(Paragraph("Calculation Steps", label_style))
        for i, step in enumerate(steps, 1):
            step_html = safe(step)
            # Bold formatting for **text**
            step_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", step_html)
            step_data = [[Paragraph(f"Step {i}: {step_html}", step_style)]]
            step_table = Table(step_data, colWidths=[doc.width])
            step_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
                                            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#bfdbfe")),
                                            ("TOPPADDING", (0, 0), (-1, -1), 8),
                                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                                            ("LEFTPADDING", (0, 0), (-1, -1), 12)]))
            story.append(step_table);
            story.append(Spacer(1, 0.2 * cm))
        story.append(Spacer(1, 0.15 * cm))

    # Results section
    if results:
        story.append(Paragraph("Final Results", label_style))
        # Handle special table data for Effect of Sample Size
        if "Sample Size Table" in results:
            table_info = results.pop("Sample Size Table")
            # First show regular results
            regular_results = {k: v for k, v in results.items()}
            if regular_results:
                result_data = [[Paragraph(f"<b>{safe(k)}:</b> {safe(v)}", result_style)] for k, v in
                               regular_results.items()]
                result_table = Table(result_data, colWidths=[doc.width])
                result_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
                                                  ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#86efac")),
                                                  ("TOPPADDING", (0, 0), (-1, -1), 10),
                                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                                                  ("LEFTPADDING", (0, 0), (-1, -1), 12)]))
                story.append(result_table);
                story.append(Spacer(1, 0.3 * cm))
            # Then show the table
            story.append(Paragraph("Sample Size Analysis Table", label_style))
            story.append(Spacer(1, 0.2 * cm))
            # Build table headers and data
            headers = list(table_info.keys())
            data_rows = list(zip(*table_info.values()))
            table_data = [[Paragraph(f"<b>{safe(h)}</b>", body_style) for h in headers]]
            for row in data_rows:
                table_data.append([Paragraph(safe(str(cell)), body_style) for cell in row])
            sample_table = Table(table_data, repeatRows=1)
            sample_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), LGRAY),
                ("GRID", (0, 0), (-1, -1), 1, BORDER),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("TOPPADDING", (0, 1), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
            ]))
            story.append(sample_table);
            story.append(Spacer(1, 0.5 * cm))
        else:
            result_data = [[Paragraph(f"<b>{safe(k)}:</b> {safe(v)}", result_style)] for k, v in results.items()]
            result_table = Table(result_data, colWidths=[doc.width])
            result_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
                                              ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#86efac")),
                                              ("TOPPADDING", (0, 0), (-1, -1), 10),
                                              ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                                              ("LEFTPADDING", (0, 0), (-1, -1), 12)]))
            story.append(result_table);
            story.append(Spacer(1, 0.5 * cm))

    # Visualization section
    if figure_bytes:
        story.append(Paragraph("Visualization", label_style))
        story.append(Spacer(1, 0.2 * cm))
        img = Image(io.BytesIO(figure_bytes), width=15 * cm, height=10 * cm)
        story.append(img)
        story.append(Spacer(1, 0.5 * cm))

    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%d %B %Y at %I:%M %p')} | "
        "National Engineering College | AI&amp;DS Department", meta_style))
    doc.build(story)
    buf.seek(0)
    return buf.read()


def show_calculation_download(topic: str, inputs: dict, steps: list, results: dict, key_suffix: str = "",
                              figure_bytes: bytes = None):
    """Show download button for manual calculation results."""
    try:
        pdf_bytes = generate_calculation_pdf(topic, inputs, steps, results, figure_bytes)
        st.download_button(
            label="📥 Download Calculation as PDF",
            data=pdf_bytes,
            file_name=f"NEC_Calc_{topic.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"calc_pdf_{key_suffix}",
        )
    except Exception as e:
        st.warning(f"PDF generation error: {e}")


# ─────────────────────────────────────────────
# Image helper
# ─────────────────────────────────────────────
def get_base64_image(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main-title{background:linear-gradient(90deg,#7c3aed 0%,#ec4899 50%,#f59e0b 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  font-size:2.8rem;font-weight:900;margin-bottom:0.2rem;line-height:1.15;}
.sub-text{font-size:1.05rem;color:#475569;margin-bottom:1rem;}
.hero-box{background:linear-gradient(135deg,#eef2ff 0%,#fdf2f8 50%,#fff7ed 100%);
  padding:1.6rem;border-radius:24px;border:2px solid #ddd6fe;
  box-shadow:0 12px 28px rgba(0,0,0,0.07);margin-bottom:1.2rem;}
.hero-heading{font-size:2rem;font-weight:900;color:#1e1b4b;margin-bottom:0.4rem;}
.hero-para{font-size:1.05rem;color:#374151;}
.coordinator-box{background:linear-gradient(135deg,#fef3c7,#fde68a,#fca5a5);color:#7c2d12;
  padding:1rem 1.3rem;border-radius:22px;border-left:10px solid #f59e0b;font-size:1.35rem;
  font-weight:900;box-shadow:0 10px 20px rgba(245,158,11,0.2);margin:1rem 0 1.2rem;text-align:center;}
.section-label{font-size:1.5rem;font-weight:800;color:#1f2937;margin:0.8rem 0;}
.dev-card{background:linear-gradient(135deg,#dbeafe,#e0f2fe,#dcfce7);padding:1.2rem;
  border-radius:22px;text-align:center;font-weight:800;font-size:1.1rem;
  box-shadow:0 10px 22px rgba(0,0,0,0.09);border:1px solid #bfdbfe;margin-bottom:0.7rem;}
.dev-reg{font-size:0.95rem;color:#334155;margin-top:0.35rem;font-weight:500;}
.year-pill{display:inline-block;background:linear-gradient(90deg,#2563eb,#7c3aed,#ec4899);
  color:white;padding:0.65rem 1.1rem;border-radius:999px;font-weight:800;margin-top:0.7rem;
  box-shadow:0 8px 18px rgba(37,99,235,0.25);}
.topic-card{background:linear-gradient(135deg,#eff6ff,#f5f3ff);padding:1rem 1.1rem;
  border-radius:18px;border:1px solid #c4b5fd;box-shadow:0 8px 18px rgba(0,0,0,0.05);
  min-height:115px;font-size:0.97rem;color:#1e1b4b;}
.info-card{background:#ffffff;padding:1.1rem;border-radius:18px;
  border:1px solid #e2e8f0;box-shadow:0 8px 18px rgba(0,0,0,0.05);margin-bottom:1rem;}
.result-pill{display:inline-block;background:linear-gradient(90deg,#10b981,#059669);
  color:#fff;padding:0.55rem 1.2rem;border-radius:999px;font-weight:800;font-size:1rem;margin-top:0.5rem;}
.result-pill-red{display:inline-block;background:linear-gradient(90deg,#ef4444,#dc2626);
  color:#fff;padding:0.55rem 1.2rem;border-radius:999px;font-weight:800;font-size:1rem;margin-top:0.5rem;}
.footer-box{margin-top:2rem;padding:0.9rem 1.1rem;border-radius:18px;
  background:linear-gradient(90deg,#eff6ff,#f5f3ff);border:1px solid #c4b5fd;
  display:flex;align-items:center;gap:14px;}
.step-row{display:flex;align-items:flex-start;gap:10px;background:#f8fafc;border-radius:12px;
  padding:0.65rem 1rem;margin-bottom:0.5rem;border-left:4px solid #7c3aed;}
.step-num{min-width:28px;height:28px;border-radius:50%;
  background:linear-gradient(135deg,#7c3aed,#ec4899);color:#fff;font-weight:800;
  font-size:0.85rem;display:flex;align-items:center;justify-content:center;}
.step-text{color:#1e293b;font-size:0.98rem;padding-top:2px;}
/* feedback */
.feedback-box{background:linear-gradient(135deg,#fdf4ff,#eff6ff);border:2px solid #ddd6fe;
  border-radius:24px;padding:1.5rem 1.8rem;margin-top:1rem;
  box-shadow:0 12px 28px rgba(0,0,0,0.06);}
.feedback-title{font-size:1.4rem;font-weight:900;color:#1e1b4b;margin-bottom:0.8rem;}
.comment-card{background:#ffffff;border-radius:16px;padding:0.9rem 1.1rem;
  margin-bottom:0.7rem;border:1px solid #e2e8f0;box-shadow:0 4px 12px rgba(0,0,0,0.04);}
.star{color:#f59e0b;font-size:1rem;}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header / Footer
# ─────────────────────────────────────────────
def show_header():
    b64 = get_base64_image(LOGO_PATH)
    if b64:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:18px;margin-bottom:12px;">
            <img src="data:image/png;base64,{b64}" width="90"
                 style="border-radius:18px;box-shadow:0 8px 16px rgba(0,0,0,0.12);">
            <div>
                <div style="font-size:1.7rem;font-weight:900;color:#111827;">
                    National Engineering College</div>
                <div style="color:#2563eb;font-weight:700;font-size:0.97rem;">
                    https://nec.edu.in</div>
            </div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("## National Engineering College\nhttps://nec.edu.in")


def show_footer():
    b64 = get_base64_image(LOGO_PATH)
    if b64:
        st.markdown(f"""<div class="footer-box">
            <img src="data:image/png;base64,{b64}" width="46" style="border-radius:12px;">
            <div><div style="font-weight:900;">National Engineering College</div>
                 <div style="color:#2563eb;font-size:0.93rem;">https://nec.edu.in</div></div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("---\n**National Engineering College**")


def section_title(t): st.markdown(f"## {t}")


def step_block(lines: list):
    for i, line in enumerate(lines, 1):
        st.markdown(f"""<div class="step-row"><div class="step-num">{i}</div>
            <div class="step-text">{line}</div></div>""", unsafe_allow_html=True)


def info_sections(explanation, example, advantages, disadvantages, applications, realtime):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("#### 📖 Explanation");
        st.write(explanation)
        st.markdown("#### 💡 Example");
        st.write(example)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("#### ✅ Advantages")
        for x in advantages: st.write(f"• {x}")
        st.markdown("#### ⚠️ Disadvantages")
        for x in disadvantages: st.write(f"• {x}")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Applications")
    cols = st.columns(len(applications))
    for col, app in zip(cols, applications):
        col.markdown(f"<div style='background:#eff6ff;border-radius:12px;padding:0.5rem 0.7rem;"
                     f"text-align:center;font-weight:600;font-size:0.92rem;color:#1d4ed8;'>"
                     f"{app}</div>", unsafe_allow_html=True)
    st.markdown("#### 🌐 Real-Time Example");
    st.info(realtime)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Code Runner
# ─────────────────────────────────────────────
def run_python_code(code_text: str, button_key: str):
    st.markdown("### 🐍 Python Code Playground")

    code = st.text_area(
        "Edit & run the code below:",
        value=code_text,
        height=200,
        key=f"code_{button_key}"
    )

    if st.button("▶ Run Python Code", key=f"run_{button_key}", use_container_width=True):
        buf = io.StringIO()

        sg = {
            "__builtins__": __builtins__,
            "math": math,
            "np": np,
            "plt": plt,
            "norm": norm,
            "t_dist": t_dist,
            "stats": stats,
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "str": str,
            "int": int,
            "float": float,
        }

        try:
            with contextlib.redirect_stdout(buf):
                exec(code, sg)

            out = buf.getvalue()

            if out.strip():
                st.code(out, language="text")
            else:
                st.success("✅ Code executed.")

        except Exception as exc:
            st.error(f"❌ Error: {exc}")

    return None


# ─────────────────────────────────────────────
# Statistical functions
# ─────────────────────────────────────────────
def z_test_known_sigma(xbar, mu, sigma, n, alpha=0.05, tail="two-tailed"):
    se = sigma / math.sqrt(n)
    z = (xbar - mu) / se
    if tail == "right-tailed":
        p, cv, rej = 1 - norm.cdf(z), norm.ppf(1 - alpha), z > norm.ppf(1 - alpha)
    elif tail == "left-tailed":
        p, cv, rej = norm.cdf(z), norm.ppf(alpha), z < norm.ppf(alpha)
    else:
        p, cv, rej = 2 * (1 - norm.cdf(abs(z))), norm.ppf(1 - alpha / 2), abs(z) > norm.ppf(1 - alpha / 2)
    return {"se": se, "z": z, "p_value": p, "critical": cv, "reject": rej}


def confidence_interval_known_sigma(xbar, sigma, n, conf=0.95):
    alpha = 1 - conf
    zs = norm.ppf(1 - alpha / 2)
    se = sigma / math.sqrt(n)
    moe = zs * se
    return {"z_star": zs, "se": se, "moe": moe,
            "lower": xbar - moe, "upper": xbar + moe}


def stratified_sample(total, groups, sample_size):
    return {name: round((size / total) * sample_size, 2) for name, size in groups}


# ─────────────────────────────────────────────
# Graphs
# ─────────────────────────────────────────────
PALETTE = {"blue": "#3b82f6", "purple": "#7c3aed", "pink": "#ec4899",
           "green": "#10b981", "orange": "#f59e0b", "red": "#ef4444", "gray": "#94a3b8"}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#f8fafc")
    ax.grid(color="#e2e8f0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#1e1b4b", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, color="#475569")
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, color="#475569")
    for sp in ax.spines.values(): sp.set_edgecolor("#e2e8f0")
    ax.tick_params(colors="#475569")


def draw_population_sample_plot(pop_size=1000, sample_size=50):
    np.random.seed(42)
    pop = np.random.normal(50, 10, pop_size)
    sample = np.random.choice(pop, size=sample_size, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#f8fafc")
    ax = axes[0]
    ax.hist(pop, bins=30, alpha=0.6, color=PALETTE["blue"], label=f"Population (n={pop_size})")
    ax.hist(sample, bins=15, alpha=0.8, color=PALETTE["orange"], label=f"Sample (n={sample_size})")
    ax.axvline(np.mean(pop), color=PALETTE["blue"], linestyle="--", linewidth=1.5,
               label=f"Pop mu={np.mean(pop):.1f}")
    ax.axvline(np.mean(sample), color=PALETTE["orange"], linestyle="--", linewidth=1.5,
               label=f"x-bar={np.mean(sample):.1f}")
    _style_ax(ax, "Population vs Sample", "Value", "Frequency");
    ax.legend(fontsize=8)
    ax2 = axes[1]
    ni = np.setdiff1d(pop, sample)
    ax2.scatter(range(len(ni[:200])), ni[:200], alpha=0.3, s=12, color=PALETTE["gray"], label="Population")
    ax2.scatter(range(len(sample)), sample, alpha=0.9, s=30, color=PALETTE["orange"], label="Sample", zorder=3)
    _style_ax(ax2, "Sample Highlighted", "Index", "Value");
    ax2.legend(fontsize=8)
    fig.tight_layout();
    return fig


def draw_random_sampling_plot(sample):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#f8fafc")
    ax = axes[0]
    ax.bar(range(1, len(sample) + 1), sample,
           color=plt.cm.viridis(np.linspace(0.2, 0.85, len(sample))),
           edgecolor="white", linewidth=0.6)
    ax.axhline(np.mean(sample), color=PALETTE["red"], linestyle="--", linewidth=1.5,
               label=f"Mean={np.mean(sample):.1f}")
    _style_ax(ax, "Random Sample Values", "Index", "Value");
    ax.legend(fontsize=9)
    axes[1].hist(sample, bins=max(5, len(sample) // 3), color=PALETTE["purple"], alpha=0.75, edgecolor="white")
    _style_ax(axes[1], "Distribution of Sample", "Value", "Frequency")
    fig.tight_layout();
    return fig


def draw_sampling_distribution(pop_mean=50.0, sigma=10.0, n=36, reps=1000):
    np.random.seed(0)
    means = [np.mean(np.random.normal(pop_mean, sigma, int(n))) for _ in range(reps)]
    tse = sigma / math.sqrt(n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#f8fafc")
    ax = axes[0]
    ax.hist(means, bins=35, density=True, alpha=0.7, color=PALETTE["blue"], edgecolor="white")
    xs = np.linspace(min(means), max(means), 300)
    ax.plot(xs, norm.pdf(xs, pop_mean, tse), color=PALETTE["red"], linewidth=2, label="Theoretical")
    ax.axvline(pop_mean, linestyle="--", linewidth=1.8, color=PALETTE["orange"], label=f"mu={pop_mean}")
    _style_ax(ax, "Sampling Distribution of x-bar", "Sample Mean", "Density");
    ax.legend(fontsize=8)
    ns = np.arange(2, max(10, int(n)) + 50)
    axes[1].plot(ns, [sigma / math.sqrt(x) for x in ns], color=PALETTE["purple"], linewidth=2)
    axes[1].scatter([n], [tse], s=100, color=PALETTE["red"], zorder=5, label=f"n={int(n)}, SE={tse:.3f}")
    _style_ax(axes[1], "SE vs n", "n", "SE");
    axes[1].legend(fontsize=8)
    fig.tight_layout();
    return fig, tse


def draw_standard_error_graph(sigma, n):
    ns = np.arange(2, max(10, int(n)) + 50)
    ses = [sigma / math.sqrt(x) for x in ns]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#f8fafc")
    se_now = sigma / math.sqrt(n)
    axes[0].plot(ns, ses, color=PALETTE["purple"], linewidth=2)
    axes[0].scatter([n], [se_now], s=100, color=PALETTE["red"], zorder=5,
                    label=f"SE at n={int(n)} -> {se_now:.4f}")
    axes[0].fill_between(ns, ses, alpha=0.12, color=PALETTE["purple"])
    _style_ax(axes[0], "SE vs Sample Size", "n", "SE");
    axes[0].legend(fontsize=9)
    ns_b = [5, 10, 20, 30, 50, 100, 200]
    ses_b = [sigma / math.sqrt(x) for x in ns_b]
    bars = axes[1].bar([str(x) for x in ns_b], ses_b,
                       color=[PALETTE["blue"]] * len(ns_b), alpha=0.8, edgecolor="white")
    for i, nb in enumerate(ns_b):
        if nb == int(n): bars[i].set_color(PALETTE["red"])
    for bar, val in zip(bars, ses_b):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    _style_ax(axes[1], f"SE by n (sigma={sigma})", "n", "SE")
    fig.tight_layout();
    return fig


def draw_hypothesis_testing_graph():
    x = np.linspace(-4.5, 4.5, 500);
    y = norm.pdf(x)
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#f8fafc")
    ax.plot(x, y, color=PALETTE["blue"], linewidth=2.5)
    crit = 1.96
    ax.fill_between(x, y, where=(x <= -crit), alpha=0.45, color=PALETTE["red"], label="Rejection Region")
    ax.fill_between(x, y, where=(x >= crit), alpha=0.45, color=PALETTE["red"])
    ax.fill_between(x, y, where=(-crit <= x) & (x <= crit), alpha=0.15, color=PALETTE["green"],
                    label="Acceptance Region")
    ax.axvline(-crit, linestyle="--", color=PALETTE["red"], linewidth=1.5)
    ax.axvline(crit, linestyle="--", color=PALETTE["red"], linewidth=1.5, label=f"CV +/-{crit}")
    ax.text(-3.5, 0.03, "Reject\nH0", ha="center", fontsize=10, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["red"], alpha=0.8))
    ax.text(3.5, 0.03, "Reject\nH0", ha="center", fontsize=10, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["red"], alpha=0.8))
    ax.text(0, 0.18, "Fail to\nReject H0", ha="center", fontsize=11, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["green"], alpha=0.8))
    _style_ax(ax, "Two-Tailed Hypothesis Testing (alpha=0.05)", "z value", "Density")
    ax.legend(fontsize=9);
    fig.tight_layout();
    return fig


def draw_z_test_curve(z_value, alpha=0.05, tail="two-tailed"):
    x = np.linspace(-4.5, 4.5, 600);
    y = norm.pdf(x)
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#f8fafc");
    ax.plot(x, y, color=PALETTE["blue"], linewidth=2.5)
    if tail == "two-tailed":
        cv = norm.ppf(1 - alpha / 2)
        ax.fill_between(x, y, where=(x <= -cv), alpha=0.4, color=PALETTE["red"], label=f"Rejection (alpha={alpha})")
        ax.fill_between(x, y, where=(x >= cv), alpha=0.4, color=PALETTE["red"])
        ax.axvline(-cv, linestyle="--", color=PALETTE["red"], linewidth=1.5, label=f"CV=+/-{cv:.3f}")
        ax.axvline(cv, linestyle="--", color=PALETTE["red"], linewidth=1.5)
    elif tail == "right-tailed":
        cv = norm.ppf(1 - alpha)
        ax.fill_between(x, y, where=(x >= cv), alpha=0.4, color=PALETTE["red"], label=f"Rejection")
        ax.axvline(cv, linestyle="--", color=PALETTE["red"], linewidth=1.5, label=f"CV={cv:.3f}")
    else:
        cv = norm.ppf(alpha)
        ax.fill_between(x, y, where=(x <= cv), alpha=0.4, color=PALETTE["red"], label=f"Rejection")
        ax.axvline(cv, linestyle="--", color=PALETTE["red"], linewidth=1.5, label=f"CV={cv:.3f}")
    in_rej = ((tail == "two-tailed" and abs(z_value) > norm.ppf(1 - alpha / 2)) or
              (tail == "right-tailed" and z_value > norm.ppf(1 - alpha)) or
              (tail == "left-tailed" and z_value < norm.ppf(alpha)))
    zc = PALETTE["red"] if in_rej else PALETTE["green"]
    ax.axvline(z_value, linewidth=2.5, color=zc, label=f"z={z_value:.3f}")
    ax.text(z_value, norm.pdf(z_value) + 0.04, "Reject H0" if in_rej else "Fail to Reject H0",
            ha="center", va="bottom", fontsize=9, color=zc, fontweight="bold")
    _style_ax(ax, f"Z-Test ({tail}, alpha={alpha})", "z value", "Density")
    ax.legend(fontsize=9);
    fig.tight_layout();
    return fig


def draw_confidence_interval_graph(xbar, lower, upper, conf=0.95):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor("#f8fafc")
    ax = axes[0];
    ax.set_facecolor("#f8fafc")
    ax.plot([lower, upper], [1, 1], linewidth=10, color=PALETTE["blue"], alpha=0.7,
            solid_capstyle="round", solid_joinstyle="round")
    ax.scatter(xbar, 1, zorder=5, s=120, color=PALETTE["orange"], label=f"x-bar={xbar:.2f}")
    ax.scatter([lower, upper], [1, 1], zorder=5, s=80, color=PALETTE["blue"], marker="|", linewidths=3)
    ax.text(lower, 1.08, f"{lower:.2f}", ha="center", fontsize=9, color="#1e1b4b")
    ax.text(upper, 1.08, f"{upper:.2f}", ha="center", fontsize=9, color="#1e1b4b")
    ax.text(xbar, 0.88, f"x-bar={xbar:.2f}", ha="center", fontsize=9, color=PALETTE["orange"], fontweight="bold")
    ax.set_ylim(0.6, 1.4);
    ax.set_yticks([])
    ax.set_title(f"{int(conf * 100)}% Confidence Interval", fontsize=12, fontweight="bold", color="#1e1b4b")
    ax.set_xlabel("Value", fontsize=10, color="#475569")
    for sp in ax.spines.values(): sp.set_edgecolor("#e2e8f0")
    ax.legend(fontsize=9)
    ax2 = axes[1];
    ax2.set_facecolor("#f8fafc")
    np.random.seed(7)
    ssig = (upper - lower) / (2 * norm.ppf(1 - (1 - conf) / 2))
    hit = 0
    for i in range(20):
        s = np.random.normal(xbar, ssig, 30)
        m = np.mean(s);
        se_s = ssig / math.sqrt(30)
        lo = m - norm.ppf(1 - (1 - conf) / 2) * se_s
        hi = m + norm.ppf(1 - (1 - conf) / 2) * se_s
        c = lo <= xbar <= hi;
        hit += c
        col = PALETTE["blue"] if c else PALETTE["red"]
        ax2.hlines(y=i, xmin=lo, xmax=hi, linewidth=3, color=col, alpha=0.6)
        ax2.scatter(m, i, s=30, color=col, zorder=3)
    ax2.axvline(xbar, color=PALETTE["orange"], linestyle="--", linewidth=1.8, label=f"True mean ({hit}/20)")
    ax2.set_yticks([]);
    ax2.set_title(f"Simulated {int(conf * 100)}% CIs", fontsize=12,
                  fontweight="bold", color="#1e1b4b")
    ax2.set_xlabel("Value", fontsize=10, color="#475569")
    for sp in ax2.spines.values(): sp.set_edgecolor("#e2e8f0")
    ax2.legend(fontsize=8);
    fig.tight_layout();
    return fig


def draw_ci_effect(sigma=10.0):
    sizes = np.array([5, 10, 20, 30, 50, 100, 200, 500])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4));
    fig.patch.set_facecolor("#f8fafc")
    axes[0].plot(sizes, [norm.ppf(0.950) * sigma / math.sqrt(n) for n in sizes], marker="o",
                 color=PALETTE["green"], label="90% CI", linewidth=2)
    axes[0].plot(sizes, [norm.ppf(0.975) * sigma / math.sqrt(n) for n in sizes], marker="s",
                 color=PALETTE["blue"], label="95% CI", linewidth=2)
    axes[0].plot(sizes, [norm.ppf(0.995) * sigma / math.sqrt(n) for n in sizes], marker="^",
                 color=PALETTE["purple"], label="99% CI", linewidth=2)
    _style_ax(axes[0], "Margin of Error vs n", "n", "MoE");
    axes[0].legend(fontsize=9)
    ns_b = [10, 25, 50, 100, 200];
    mv = [norm.ppf(0.975) * sigma / math.sqrt(n) for n in ns_b]
    bars = axes[1].bar([str(n) for n in ns_b], mv, color=[PALETTE["blue"]] * 5, alpha=0.8, edgecolor="white")
    for bar, val in zip(bars, mv):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", fontsize=9, color="#1e1b4b")
    _style_ax(axes[1], f"95% MoE (sigma={sigma})", "n", "MoE")
    fig.tight_layout();
    return fig


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

STATS_SYSTEM_PROMPT = """You are an expert statistics tutor specializing ONLY in Inferential Statistics.

If the question is NOT related to: Population/Sample, Random Sampling, Sampling Distribution,
Standard Error, Hypothesis Testing, Z-Test, Confidence Intervals, Effect of Sample Size —
respond ONLY: "Please ask a question related to Inferential Statistics."

STRICT FORMATTING RULES:
1. NEVER use LaTeX. No fractions with backslash, no dollar signs.
2. Plain text math only: a/b for fractions, sqrt(n) for roots, x for multiply, +/- for plus-minus.
3. Number every calculation step.
4. Use **bold** for key terms and final answers.
5. Label the final answer clearly.
6. Simple, student-friendly language."""


def ask_llm(question: str, topic_context: str = "") -> str:
    msg = f"[Topic: {topic_context}]\n\n{question}" if topic_context else question
    try:
        r = requests.post(OPENROUTER_URL, timeout=30,
                          headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                                   "Content-Type": "application/json",
                                   "HTTP-Referer": "https://nec.edu.in",
                                   "X-Title": "NEC Inferential Statistics App"},
                          json={"model": OPENROUTER_MODEL, "max_tokens": 1024, "temperature": 0.3,
                                "messages": [{"role": "system", "content": STATS_SYSTEM_PROMPT},
                                             {"role": "user", "content": msg}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"⚠️ API error: {e.response.status_code} – {e.response.text}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"


def clean_llm_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"\\\[|\\\]|\$\$", "", text)
    text = re.sub(r"\\\(|\\\)", "", text)
    text = re.sub(r"(?<!\$)\$(?!\$)", "", text)
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\(?:display|text|script)style\s*", "", text)
    for pat, rep in [("\\\\times", "x"), ("\\\\cdot", "·"), ("\\\\div", "/"), ("\\\\pm", "+/-"),
                     ("\\\\leq", "<="), ("\\\\geq", ">="), ("\\\\neq", "!="), ("\\\\approx", "~="),
                     ("\\\\mu", "mu"), ("\\\\sigma", "sigma"), ("\\\\alpha", "alpha"),
                     ("\\\\beta", "beta"), ("\\\\chi", "chi")]:
        text = re.sub(pat, rep, text)
    text = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\s+(\S+)", r"sqrt(\1)", text)
    text = re.sub(r"\\bar\{([^}]+)\}", r"\1-bar", text)
    text = re.sub(r"_\{([^}]+)\}", r"_\1", text)
    text = re.sub(r"\^\{([^}]+)\}", r"^\1", text)
    text = re.sub(r"\\math(?:bf|it|rm|cal|bb)\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\(?:left|right)\s*[\(\)\[\]\{\}|]", lambda m: m.group(0)[-1], text)
    text = re.sub(r"\\(?:left|right)\.", "", text)
    text = re.sub(r"\\[,!:;]", " ", text)
    text = re.sub(r"\\(?:quad|qquad)\b", "  ", text)
    text = re.sub(r"\\([A-Za-z]+)\b", r"\1", text)
    text = re.sub(r"(?<![a-zA-Z0-9_])[{}](?![a-zA-Z0-9_])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_llm_answer(answer: str, topic: str = "", question: str = "", key_suffix: str = ""):
    cleaned = clean_llm_output(answer)
    st.markdown("""
    <div style="background:linear-gradient(135deg,#f0fdf4,#eff6ff);border:1.5px solid #86efac;
         border-radius:18px;padding:1.3rem 1.5rem;margin-top:0.8rem;
         box-shadow:0 8px 20px rgba(0,0,0,0.06);">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.7rem;">
        <span style="font-size:1.3rem;">🤖</span>
        <span style="font-weight:800;color:#166534;font-size:1rem;">AI Tutor Answer &nbsp;

    </div>""", unsafe_allow_html=True)
    st.markdown(cleaned)
    st.markdown("</div>", unsafe_allow_html=True)
    # PDF download
    if topic and question and cleaned and not cleaned.startswith("⚠️"):
        show_pdf_download(topic, question, cleaned, key_suffix=key_suffix)


def home_question_answer(question: str):
    if not question.strip():
        st.warning("Please enter a question.");
        return
    with st.spinner("🤖 AI Tutor is thinking…"):
        answer = ask_llm(question, "Inferential Statistics")
    render_llm_answer(answer, "Inferential Statistics", question, "home")


def solve_text_question(question: str, topic_name: str):
    if not question.strip():
        st.warning("Please enter a question.");
        return
    with st.spinner(f"🤖 Solving with AI Tutor ({topic_name})…"):
        answer = ask_llm(question, topic_name)
    render_llm_answer(answer, topic_name, question, topic_name.replace(" ", "_"))


# ─────────────────────────────────────────────
# Feedback Widget
# ─────────────────────────────────────────────
def show_feedback_section():
    fb = _load_feedback()
    likes = fb.get("likes", 0)

    st.markdown("<div class='feedback-title'>💬 Feedback &amp; Reviews</div>",
                unsafe_allow_html=True)

    # Like button
    col_like, col_count, _ = st.columns([1, 1.2, 4])
    with col_like:
        already_liked = st.session_state.get("liked", False)
        label = "✅ Liked!" if already_liked else "👍 Like"
        if st.button(label, key="like_btn", use_container_width=True,
                     disabled=already_liked):
            add_like()
            st.session_state.liked = True
            st.rerun()
    with col_count:
        fresh_likes = _load_feedback().get("likes", 0)
        st.markdown(
            f"<div style='padding-top:0.45rem;font-size:1.1rem;font-weight:800;color:#7c3aed;'>"
            f"❤️ {fresh_likes:,} Likes</div>",
            unsafe_allow_html=True)

    st.markdown("---")

    # Add review
    st.markdown("#### ✍️ Leave a Review")
    c1, c2 = st.columns([2, 1])
    with c1:
        name = st.text_input("Your name (optional)", placeholder="e.g. Harini", key="fb_name")
        comment = st.text_area("Your feedback", height=90, key="fb_comment",
                               placeholder="Write your thoughts about this app…")
    with c2:
        rating = st.select_slider("Rating", options=[1, 2, 3, 4, 5], value=5,
                                  format_func=lambda x: "⭐" * x, key="fb_rating")

    if st.button("📨 Submit Review", key="fb_submit", use_container_width=True):
        if comment.strip():
            add_comment(name, comment, rating)
            st.success("🎉 Thank you for your feedback!")
            st.rerun()
        else:
            st.warning("Please write something before submitting.")

    # Show reviews
    all_comments = list(reversed(_load_feedback().get("comments", [])))
    if all_comments:
        st.markdown(f"#### 🗣️ Reviews ({len(all_comments)})")
        for c in all_comments:
            stars = "⭐" * c.get("rating", 5)
            st.markdown(f"""
            <div class="comment-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="font-weight:800;color:#1e1b4b;">
                        👤 {c.get('name', 'Anonymous')}</div>
                    <div style="font-size:0.82rem;color:#94a3b8;">{c.get('time', '')}</div>
                </div>
                <div class="star" style="margin:0.3rem 0 0.4rem;">{stars}</div>
                <div style="color:#374151;font-size:0.95rem;">{c.get('comment', '')}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No reviews yet. Be the first to leave feedback! 🌟")

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#   A P P   B O O T S T R A P
# ═══════════════════════════════════════════════════════
inject_custom_css()
_vc = increment_visitor()
show_header()

st.markdown("<div class='main-title'>Inferential Statistics Learning App</div>",
            unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Learn concepts, solve problems step-by-step, "
            "run Python code live, and explore interactive graphs – all in one place.</div>",
            unsafe_allow_html=True)

# ─── Sidebar ───
TOPIC_OPTIONS = ["Home", "Population vs Sample", "Random Sampling", "Sampling Distribution",
                 "Standard Error of Mean", "Hypothesis Testing", "Z-Test Solver",
                 "Confidence Interval", "Effect of Sample Size"]
ICONS = {"Home": "🏠", "Population vs Sample": "👥", "Random Sampling": "🎲",
         "Sampling Distribution": "📊", "Standard Error of Mean": "📏",
         "Hypothesis Testing": "🔬", "Z-Test Solver": "⚗️",
         "Confidence Interval": "📐", "Effect of Sample Size": "📈"}

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = "Home"

st.sidebar.markdown(f"""
<div style="background:linear-gradient(135deg,#1e1b4b,#3730a3);border-radius:16px;
     padding:0.85rem 1rem;margin-bottom:1rem;text-align:center;
     box-shadow:0 6px 16px rgba(0,0,0,0.2);">
    <div style="color:#a5b4fc;font-size:0.75rem;font-weight:600;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">
        👁️ Visitor Stats</div>
    <div style="display:flex;justify-content:space-around;align-items:center;">
        <div><div style="color:#fff;font-size:1.6rem;font-weight:900;">{_vc.get('total', 0):,}</div>
             <div style="color:#c7d2fe;font-size:0.72rem;">Total Visits</div></div>
        <div style="width:1px;height:36px;background:#4f46e5;"></div>
        <div><div style="color:#34d399;font-size:1.6rem;font-weight:900;">{_vc.get('today', 0):,}</div>
             <div style="color:#c7d2fe;font-size:0.72rem;">Today</div></div>
    </div></div>""", unsafe_allow_html=True)

st.sidebar.markdown("## 🗂️ Topic Navigator")
search = st.sidebar.text_input("🔍 Search topic or keyword")
filtered = [t for t in TOPIC_OPTIONS if search.lower() in t.lower()] if search else TOPIC_OPTIONS

for t in filtered:
    active = st.session_state.selected_topic == t
    if st.sidebar.button(f"{ICONS.get(t, '')} {t}", use_container_width=True,
                         type="primary" if active else "secondary", key=f"nav_{t}"):
        st.session_state.selected_topic = t
        st.rerun()

menu = st.session_state.selected_topic

# ═══════════════════════════════════════════════════════
#   H O M E
# ═══════════════════════════════════════════════════════
if menu == "Home":
    st.markdown("""<div class="hero-box">
        <div class="hero-heading">🎓 Inferential Statistics Smart Learning App</div>
        <div class="hero-para">Understand inferential statistics from the ground up. Covers
        <strong>Population vs Sample, Random Sampling, Sampling Distribution,
        Standard Error, Hypothesis Testing, Z-Test, Confidence Intervals</strong>
        and <strong>Effect of Sample Size</strong> – with step-by-step solutions,
        live Python code, and interactive graphs.</div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="coordinator-box">
        🌟 Course Instructor: Dr. J. Naskath, Asso. Prof / AI&amp;DS 🌟
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>👩‍💻 Developed By</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("<div class='dev-card'>S. HARINI<div class='dev-reg'>Reg. No: 24243060</div></div>",
                    unsafe_allow_html=True)
    with d2:
        st.markdown("<div class='dev-card'>S. SWETHA<div class='dev-reg'>Reg. No: 24243048</div></div>",
                    unsafe_allow_html=True)
    with d3:
        st.markdown("<div class='dev-card'>M. GOWRI<div class='dev-reg'>Reg. No: 24243004</div></div>",
                    unsafe_allow_html=True)
    st.markdown("<div class='year-pill'>2ND YEAR – AI &amp; DS</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔎 Ask Any Inferential Statistics Question")
    q_home = st.text_area("Type your question here:", height=100,
                          placeholder="Example: A school has 500 students (300 juniors, 200 seniors). "
                                      "How many should be in a sample of 100 using stratified sampling?")
    if st.button("💡 Answer My Question", use_container_width=True):
        home_question_answer(q_home)

    st.markdown("---")
    st.markdown("### 🎯 Topic Cards")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("<div class='topic-card'><b>👥 Population / Sample</b><br><br>"
                    "Understand populations, samples, and why they matter in statistics.</div>", unsafe_allow_html=True)
    with tc2:
        st.markdown("<div class='topic-card'><b>🔬 Hypothesis Testing / Z-Test</b><br><br>"
                    "Formulate H0 and H1, compute z-statistics, find p-values and decisions.</div>",
                    unsafe_allow_html=True)
    with tc3:
        st.markdown("<div class='topic-card'><b>📐 Confidence Intervals / SE</b><br><br>"
                    "Estimate population mean range, compute margin of error and precision.</div>",
                    unsafe_allow_html=True)

    # ── FEEDBACK SECTION ──
    st.markdown("---")
    show_feedback_section()


# ═══════════════════════════════════════════════════════
#   P O P U L A T I O N   V S   S A M P L E
# ═══════════════════════════════════════════════════════
elif menu == "Population vs Sample":
    section_title("👥 Population vs Sample")
    info_sections(
        "A **population** is the complete set. A **sample** is a subset drawn from it.",
        "All 10,000 university students = population. 200 selected students = sample.",
        ["Faster and cheaper", "Feasible for large populations", "Can yield accurate estimates"],
        ["May not perfectly represent population", "Sampling bias possible"],
        ["Surveys", "Medical Studies", "Quality Control", "Education Research"],
        "E-commerce company surveys 500 of 2 million customers to understand behaviour."
    )
    st.markdown("---");
    st.markdown("### 🧮 Manual Solver")
    c1, c2 = st.columns(2)
    with c1:
        pop_size = st.slider("Population size (N)", 200, 10000, 1000, step=100)
    with c2:
        sample_size = st.slider("Sample size (n)", 10, 500, 50, step=10)
    st.info(f"**N={pop_size}** | **n={sample_size}** | Fraction={sample_size / pop_size * 100:.2f}%")
    fig_pop = draw_population_sample_plot(pop_size, sample_size)
    st.pyplot(fig_pop)
    # Capture figure bytes for PDF
    buf_pop = io.BytesIO()
    fig_pop.savefig(buf_pop, format='png', dpi=150, bbox_inches='tight')
    buf_pop.seek(0)
    fig_bytes_pop = buf_pop.getvalue()
    # PDF Download for Manual Calculation
    show_calculation_download(
        topic="Population vs Sample",
        inputs={"Population size (N)": pop_size, "Sample size (n)": sample_size},
        steps=[f"Fraction = n/N = {sample_size}/{pop_size} = {sample_size / pop_size:.4f}",
               f"Percentage = {sample_size / pop_size * 100:.2f}%"],
        results={"Fraction": f"{sample_size / pop_size:.4f}", "Percentage": f"{sample_size / pop_size * 100:.2f}%"},
        key_suffix="pop_sample",
        figure_bytes=fig_bytes_pop
    )
    st.markdown("---");
    st.markdown("### ❓ Solve a Text Question")
    q_ps = st.text_area("Enter your question:", key="q_pop")
    if st.button("Solve", key="solve_pop", use_container_width=True):
        solve_text_question(q_ps, "Population vs Sample")

    run_python_code("""import numpy as np
pop=np.random.normal(50,10,1000); s=np.random.choice(pop,50,replace=False)
print(f"Pop mean: {np.mean(pop):.4f}  Sample mean: {np.mean(s):.4f}")
print(f"Pop std:  {np.std(pop):.4f}  Sample std:  {np.std(s,ddof=1):.4f}")
print(f"Fraction: {len(s)/len(pop)*100:.1f}%")
""", "popsample")


# ═══════════════════════════════════════════════════════
#   R A N D O M   S A M P L I N G
# ═══════════════════════════════════════════════════════
elif menu == "Random Sampling":
    section_title("🎲 Random Sampling")
    info_sections(
        "**Random sampling** gives every member equal probability of selection.",
        "From 100 students, randomly select 10. Each has 10% chance.",
        ["Eliminates bias", "Representative sample", "Statistically valid"],
        ["Needs complete population list", "Can still be unrepresentative by chance"],
        ["Election Polls", "Hospital Studies", "Quality Inspections", "Market Research"],
        "Hospital randomly selects 30 patients to assess treatment satisfaction."
    )
    st.markdown("---");
    st.markdown("### 🧮 Manual Solver")
    c1, c2 = st.columns(2)
    with c1:
        pop_range = st.slider("Population range (1 to N)", 20, 500, 100)
    with c2:
        n_sample = st.slider("Sample size (n)", 3, min(pop_range, 50), 10)
    np.random.seed(None)
    cs = np.sort(np.random.choice(range(1, pop_range + 1), size=n_sample, replace=False))
    st.success(f"**Random sample:** {cs.tolist()}")
    ca, cb, cc = st.columns(3)
    ca.metric("Mean", f"{np.mean(cs):.2f}")
    cb.metric("Median", f"{np.median(cs):.2f}")
    cc.metric("Std Dev", f"{np.std(cs, ddof=1):.2f}")
    fig_rs = draw_random_sampling_plot(cs)
    st.pyplot(fig_rs)
    # Capture figure bytes for PDF
    buf_rs = io.BytesIO()
    fig_rs.savefig(buf_rs, format='png', dpi=150, bbox_inches='tight')
    buf_rs.seek(0)
    fig_bytes_rs = buf_rs.getvalue()
    # PDF Download for Manual Calculation
    show_calculation_download(
        topic="Random Sampling",
        inputs={"Population range": f"1 to {pop_range}", "Sample size (n)": n_sample},
        steps=[f"Random sample generated: {cs.tolist()}",
               f"Mean = {np.mean(cs):.4f}",
               f"Median = {np.median(cs):.4f}",
               f"Std Dev = {np.std(cs, ddof=1):.4f}"],
        results={"Mean": f"{np.mean(cs):.4f}", "Median": f"{np.median(cs):.4f}",
                 "Std Dev": f"{np.std(cs, ddof=1):.4f}"},
        key_suffix="random_sampling",
        figure_bytes=fig_bytes_rs
    )
    st.markdown("---");
    st.markdown("### Stratified Sampling Calculator")
    g1, g2 = st.columns(2)
    with g1:
        gnames = st.text_input("Group names (comma-separated)", value="Juniors,Seniors")
        gsizes = st.text_input("Group sizes (comma-separated)", value="300,200")
    with g2:
        tpop = st.number_input("Total population", value=500, min_value=1)
        ssz = st.number_input("Total sample size", value=100, min_value=1)
    if st.button("Calculate Stratified Sample", use_container_width=True):
        try:
            ns = [n.strip() for n in gnames.split(",")]
            sz = [int(s.strip()) for s in gsizes.split(",")]
            assert len(ns) == len(sz)
            res = stratified_sample(tpop, list(zip(ns, sz)), ssz)
            for nm, val in res.items():
                st.info(f"**{nm}:** ({dict(zip(ns, sz))[nm]}/{tpop}) x {ssz} = **{val:.1f} ≈ {round(val)}**")
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("---");
    st.markdown("### ❓ Solve a Text Question")
    q_rs = st.text_area("Enter your question:", key="q_random")
    if st.button("Solve", key="solve_random", use_container_width=True):
        solve_text_question(q_rs, "Random Sampling")
    run_python_code("""import numpy as np
s=np.random.choice(range(1,101),size=10,replace=False)
print("SRS Sample:", sorted(s))
t=500;j=300;sr=200;n=100
print(f"Stratified - Juniors:{(j/t)*n:.0f}, Seniors:{(sr/t)*n:.0f}")
""", "random")


# ═══════════════════════════════════════════════════════
#   S A M P L I N G   D I S T R I B U T I O N
# ═══════════════════════════════════════════════════════
elif menu == "Sampling Distribution":
    section_title("📊 Sampling Distribution")
    info_sections(
        "Distribution of x-bar from repeated samples. By CLT: approximately N(mu, sigma^2/n) for large n.",
        "Population N(mu=50,sigma=10), n=36. 1000 sample means ~ N(50,1.667).",
        ["Foundation of inference", "CLT applies even to non-normal populations"],
        ["Requires large n for skewed populations", "Assumes independence"],
        ["Research Design", "Quality Control", "Actuarial Science", "ML"],
        "Teacher draws many groups of 36 students, records each group average."
    )
    st.markdown("---");
    st.markdown("### 🧮 Manual Solver")
    c1, c2 = st.columns(2)
    with c1:
        pm = st.number_input("Population mean (mu)", value=50.0, key="sd_mu")
        sg = st.number_input("Population std dev (sigma)", value=10.0, min_value=0.01, key="sd_sigma")
    with c2:
        ns = st.number_input("Sample size (n)", value=36, min_value=1, key="sd_n")
        rp = st.slider("Repetitions", 100, 10000, 1000)
    tse = sg / math.sqrt(ns)
    ca, cb, cc = st.columns(3)
    ca.metric("mu", f"{pm:.4f}");
    cb.metric("SE=sigma/sqrt(n)", f"{tse:.4f}")
    cc.metric("Sampling Dist.", f"N({pm},{tse:.3f}^2)")
    if st.button("Run Simulation", use_container_width=True, key="run_sd"):
        fig_sd, se_sim = draw_sampling_distribution(pm, sg, int(ns), rp)
        step_block([f"Population N(mu={pm},sigma={sg})", f"n={int(ns)}",
                    f"SE=sigma/sqrt(n)={sg}/sqrt({int(ns)})={tse:.4f}",
                    f"Simulated SE~{se_sim:.4f}"])
        st.pyplot(fig_sd)
        # Capture figure bytes for PDF
        buf_sd = io.BytesIO()
        fig_sd.savefig(buf_sd, format='png', dpi=150, bbox_inches='tight')
        buf_sd.seek(0)
        fig_bytes_sd = buf_sd.getvalue()
        # PDF Download for Manual Calculation
        show_calculation_download(
            topic="Sampling Distribution",
            inputs={"Population mean (μ)": pm, "Population SD (σ)": sg, "Sample size (n)": int(ns), "Repetitions": rp},
            steps=[f"Population N(mu={pm}, sigma={sg})",
                   f"Sample size n = {int(ns)}",
                   f"SE = σ/√n = {sg}/√{int(ns)} = {tse:.4f}",
                   f"Simulated SE = {se_sim:.4f}"],
            results={"Theoretical SE": f"{tse:.4f}", "Simulated SE": f"{se_sim:.4f}",
                     "Sampling Distribution": f"N({pm}, {tse:.3f}²)"},
            key_suffix="sampling_dist",
            figure_bytes=fig_bytes_sd
        )
    st.markdown("---");
    st.markdown("### ❓ Solve a Text Question")
    q_sd = st.text_area("Enter your question:", key="q_sd")
    if st.button("Solve", key="solve_sd", use_container_width=True):
        solve_text_question(q_sd, "Sampling Distribution")
    run_python_code("""import numpy as np,math
mu,sg,n,rp=50,10,36,1000
means=[np.mean(np.random.normal(mu,sg,n)) for _ in range(rp)]
print(f"Theoretical SE:{sg/math.sqrt(n):.4f}  Simulated SE:{np.std(means,ddof=1):.4f}")
print(f"Mean of means:{sum(means)/len(means):.4f}  (should ~ {mu})")
""", "samplingdist")


# ═══════════════════════════════════════════════════════
#   S T A N D A R D   E R R O R
# ═══════════════════════════════════════════════════════
elif menu == "Standard Error of Mean":
    section_title("📏 Standard Error of Mean")
    info_sections(
        "SE = sigma / sqrt(n). How much x-bar varies across repeated samples.",
        "sigma=12, n=36. SE=12/sqrt(36)=12/6=**2**.",
        ["Simple formula", "Used in CIs and Z-tests", "Shows precision improvement with n"],
        ["Requires known sigma", "False confidence if assumptions violated"],
        ["Confidence Intervals", "Hypothesis Testing", "Research Papers", "Clinical Trials"],
        "Factory: sigma=5ml, n=100. SE=5/sqrt(100)=0.5ml."
    )
    st.markdown("---");
    st.markdown("### 🧮 Manual Solver")
    c1, c2 = st.columns(2)
    with c1:
        sg_se = st.number_input("sigma", value=12.0, min_value=0.001, key="se_sigma")
    with c2:
        n_se = st.number_input("n", value=36, min_value=1, key="se_n")
    se_v = sg_se / math.sqrt(n_se)
    step_block([f"sigma={sg_se}, n={int(n_se)}",
                f"SE=sigma/sqrt(n)={sg_se}/sqrt({int(n_se)})={sg_se}/{math.sqrt(n_se):.4f}",
                f"SE=**{se_v:.4f}**"])
    ca, cb = st.columns(2)
    ca.metric("SE", f"{se_v:.4f}");
    cb.metric("Interpretation", f"x-bar varies +/-{se_v:.4f} from mu")
    fig_se = draw_standard_error_graph(sg_se, n_se)
    st.pyplot(fig_se)
    # Capture figure bytes for PDF
    buf_se = io.BytesIO()
    fig_se.savefig(buf_se, format='png', dpi=150, bbox_inches='tight')
    buf_se.seek(0)
    fig_bytes_se = buf_se.getvalue()
    # PDF Download for Manual Calculation
    show_calculation_download(
        topic="Standard Error of Mean",
        inputs={"Population SD (σ)": sg_se, "Sample size (n)": n_se},
        steps=[f"σ = {sg_se}, n = {int(n_se)}",
               f"SE = σ/√n = {sg_se}/√{int(n_se)} = {sg_se}/{math.sqrt(n_se):.4f}",
               f"SE = {se_v:.4f}"],
        results={"Standard Error (SE)": f"{se_v:.4f}", "Interpretation": f"x-bar varies ±{se_v:.4f} from μ"},
        key_suffix="se_mean",
        figure_bytes=fig_bytes_se
    )
    st.markdown("---");
    st.markdown("### ❓ Solve a Text Question")
    q_se = st.text_area("Enter your question:", key="q_se")
    if st.button("Solve", key="solve_se", use_container_width=True):
        solve_text_question(q_se, "Standard Error of Mean")
    run_python_code("""import math
sg=12;n=36;SE=sg/math.sqrt(n)
print(f"SE={sg}/sqrt({n})={SE:.4f}")
for nv in [4,9,16,25,36,49,64,100]:
    print(f"  n={nv:4d}  SE={sg/math.sqrt(nv):.4f}")
""", "se")


# ═══════════════════════════════════════════════════════
#   H Y P O T H E S I S   T E S T I N G
# ═══════════════════════════════════════════════════════
elif menu == "Hypothesis Testing":
    section_title("🔬 Hypothesis Testing")
    info_sections(
        "Formal procedure to decide if sample data provides enough evidence to reject H0.",
        "H0: mu=50 | H1: mu!=50. alpha=0.05. If |z|>1.96 -> Reject H0.",
        ["Objective and reproducible", "Quantifies uncertainty via p-values"],
        ["Does not prove H0", "Sensitive to sample size", "p-value often misinterpreted"],
        ["Medical Trials", "Manufacturing", "A/B Testing", "Academic Research"],
        "School checks if new teaching method improved marks significantly."
    )
    st.markdown("---");
    st.markdown("### 📋 Hypothesis Testing Framework")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Step-by-step procedure:**")
        step_block(["State H0 (equality / no effect)",
                    "State H1 (inequality / effect)",
                    "Choose alpha (0.01, 0.05, or 0.10)",
                    "Select test (Z-test, t-test, chi-sq...)",
                    "Compute test statistic",
                    "Find p-value OR compare to critical value",
                    "Decision: Reject H0 if p-value < alpha"])
    with c2:
        st.markdown("**Critical values (two-tailed):**")
        st.table({"alpha": ["0.10", "0.05", "0.01", "0.001"],
                  "Critical z": ["+-1.645", "+-1.960", "+-2.576", "+-3.291"],
                  "Confidence": ["90%", "95%", "99%", "99.9%"]})
    st.markdown("### 📊 Visualisation")
    st.pyplot(draw_hypothesis_testing_graph())
    st.markdown("---");
    st.markdown("### ❓ Solve a Text Question")
    q_ht = st.text_area("Enter your question:", key="q_ht")
    if st.button("Solve", key="solve_ht", use_container_width=True):
        solve_text_question(q_ht, "Hypothesis Testing")
    run_python_code("""from scipy.stats import norm
xb=53;mu=50;sg=10;n=36;a=0.05
SE=sg/(n**0.5);z=(xb-mu)/SE;p=2*(1-norm.cdf(abs(z)));cv=norm.ppf(1-a/2)
print(f"SE={SE:.4f}  z={z:.4f}  p={p:.6f}  CV=+-{cv:.4f}")
print("Decision:","Reject H0" if abs(z)>cv else "Fail to Reject H0")
""", "ht")


# ═══════════════════════════════════════════════════════
#   Z - T E S T   S O L V E R
# ═══════════════════════════════════════════════════════
elif menu == "Z-Test Solver":
    section_title("⚗️ Z-Test Solver")
    info_sections(
        "Z-test when sigma is known. Formula: z=(x-bar - mu)/(sigma/sqrt(n)).",
        "mu=75,n=36,x-bar=78,sigma=12. z=1.50. CV=+-1.96. Fail to Reject H0.",
        ["Simple and fast", "Exact when sigma known"],
        ["Requires known sigma", "Needs n>=30 or normal population"],
        ["Exam Analysis", "Factory Testing", "Business KPIs", "Medical Research"],
        "Bottled water: claim 500ml. n=64, x-bar=497ml, sigma=8ml."
    )
    st.markdown("---");
    st.markdown("### 🧮 Manual Solver")
    c1, c2 = st.columns(2)
    with c1:
        xbz = st.number_input("Sample mean (x-bar)", value=78.0, key="z_xbar")
        muz = st.number_input("Population mean (mu, H0)", value=75.0, key="z_mu")
        sgz = st.number_input("Population std dev (sigma)", value=12.0, min_value=0.001, key="z_sigma")
    with c2:
        nz = st.number_input("Sample size (n)", value=36, min_value=1, key="z_n")
        az = st.number_input("Significance level (alpha)", value=0.05, min_value=0.001,
                             max_value=0.5, step=0.01, key="z_alpha")
        tailz = st.selectbox("Test type", ["two-tailed", "right-tailed", "left-tailed"], key="z_tail")
    res = z_test_known_sigma(xbz, muz, sgz, int(nz), az, tailz)
    if st.button("⚗️ Solve Z-Test", use_container_width=True, key="btn_solve_ztest"):
        h1sym = "!=" if tailz == "two-tailed" else (">" if tailz == "right-tailed" else "<")
        step_block([f"H0: mu={muz}  |  H1: mu {h1sym} {muz}",
                    f"SE=sigma/sqrt(n)={sgz}/sqrt({int(nz)})={res['se']:.4f}",
                    f"z=(x-bar - mu)/SE=({xbz}-{muz})/{res['se']:.4f}=**{res['z']:.4f}**",
                    f"P-value={res['p_value']:.6f}",
                    f"Critical value ({tailz}, alpha={az})={res['critical']:.4f}",
                    f"|z|={abs(res['z']):.4f} {'>' if res['reject'] else '<='} {abs(res['critical']):.4f}"])
        if res["reject"]:
            st.markdown("<div class='result-pill-red'>❌ Reject H0 (Statistically Significant)</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-pill'>✅ Fail to Reject H0 (Not Statistically Significant)</div>",
                        unsafe_allow_html=True)
    st.markdown("### 📊 Z-Test Curve")
    fig_z = draw_z_test_curve(res["z"], az, tailz)
    st.pyplot(fig_z)
    # Capture figure bytes for PDF
    buf_z = io.BytesIO()
    fig_z.savefig(buf_z, format='png', dpi=150, bbox_inches='tight')
    buf_z.seek(0)
    fig_bytes_z = buf_z.getvalue()
    ca, cb, cc, cd = st.columns(4)
    ca.metric("z", f"{res['z']:.4f}");
    cb.metric("SE", f"{res['se']:.4f}")
    cc.metric("p-value", f"{res['p_value']:.6f}");
    cd.metric("CV", f"{abs(res['critical']):.4f}")
    show_calculation_download(
        topic="Z-Test Solver",
        inputs={"Sample mean (x-bar)": xbz, "Population mean (mu)": muz, "Population SD (sigma)": sgz,
                "Sample size (n)": int(nz), "Alpha": az, "Test type": tailz},
        steps=["H0: mu={} | H1: mu {} {}".format(muz, "!=" if tailz == "two-tailed" else (
            ">" if tailz == "right-tailed" else "<"), muz),
               "SE = sigma/sqrt(n) = {}/sqrt({}) = {:.4f}".format(sgz, int(nz), res['se']),
               "z = (x-bar - mu)/SE = ({} - {})/{:.4f} = {:.4f}".format(xbz, muz, res['se'], res['z']),
               "P-value = {:.6f}".format(res['p_value']),
               "Critical value ({}, alpha={}) = {:.4f}".format(tailz, az, res['critical'])],
        results={"z-statistic": "{:.4f}".format(res['z']), "Standard Error (SE)": "{:.4f}".format(res['se']),
                 "P-value": "{:.6f}".format(res['p_value']), "Critical Value": "{:.4f}".format(abs(res['critical'])),
                 "Decision": "Reject H0" if res['reject'] else "Fail to Reject H0"},
        key_suffix="z_test",
        figure_bytes=fig_bytes_z
    )
    st.markdown("---");
    st.markdown("### ❓ Solve a Text Question")
    q_z = st.text_area("Enter your question:", key="q_z")
    if st.button("Solve", key="solve_z", use_container_width=True):
        solve_text_question(q_z, "Z-Test Solver")
    run_python_code(f"""import math
from scipy.stats import norm
xb={xbz};mu={muz};sg={sgz};n={int(nz)};a={az};tail="{tailz}"
SE=sg/math.sqrt(n);z=(xb-mu)/SE
if tail=="right-tailed": p=1-norm.cdf(z);cv=norm.ppf(1-a);rej=z>cv
elif tail=="left-tailed": p=norm.cdf(z);cv=norm.ppf(a);rej=z<cv
else: p=2*(1-norm.cdf(abs(z)));cv=norm.ppf(1-a/2);rej=abs(z)>cv
print(f"SE={{SE:.4f}}  z={{z:.4f}}  p={{p:.6f}}  CV={{cv:.4f}}")
print("Decision:","Reject H0" if rej else "Fail to Reject H0")
""", "ztest")


# ═══════════════════════════════════════════════════════
#   C O N F I D E N C E   I N T E R V A L
# ═══════════════════════════════════════════════════════
elif menu == "Confidence Interval":
    section_title("📐 Confidence Interval")
    info_sections(
        "CI = x-bar +/- z* x (sigma/sqrt(n)). For 95% CI: z*=1.96.",
        "x-bar=52,sigma=10,n=64. SE=1.25, MoE=2.45. CI=(49.55,54.45).",
        ["Range estimate", "Communicates precision", "Standard in reporting"],
        ["Requires known sigma", "'95% confident' often misinterpreted"],
        ["Medical Research", "Survey Reporting", "Quality Assurance", "Economic Forecasting"],
        "Company estimates average spend Rs.1250 with 95% CI (Rs.1180, Rs.1320)."
    )
    st.markdown("---");
    st.markdown("### 🧮 Manual Solver")
    c1, c2 = st.columns(2)
    with c1:
        xbc = st.number_input("Sample mean (x-bar)", value=52.0, key="ci_xbar")
        sgc = st.number_input("Population std dev (sigma)", value=10.0, min_value=0.001, key="ci_sigma")
    with c2:
        nc = st.number_input("Sample size (n)", value=64, min_value=1, key="ci_n")
        conf = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="ci_conf")
    ci = confidence_interval_known_sigma(xbc, sgc, int(nc), conf)
    if st.button("📐 Compute Confidence Interval", use_container_width=True, key="btn_solve_ci"):
        step_block([f"x-bar={xbc}, sigma={sgc}, n={int(nc)}, Confidence={int(conf * 100)}%",
                    f"alpha=1-{conf}={1 - conf:.2f}  ->  alpha/2={(1 - conf) / 2:.4f}",
                    f"z*=norm.ppf(1-alpha/2)=**{ci['z_star']:.4f}**",
                    f"SE=sigma/sqrt(n)={sgc}/sqrt({int(nc)})=**{ci['se']:.4f}**",
                    f"MoE=z* x SE={ci['z_star']:.4f} x {ci['se']:.4f}=**{ci['moe']:.4f}**",
                    f"Lower=x-bar-MoE={xbc}-{ci['moe']:.4f}=**{ci['lower']:.4f}**",
                    f"Upper=x-bar+MoE={xbc}+{ci['moe']:.4f}=**{ci['upper']:.4f}**",
                    f"**{int(conf * 100)}% CI=({ci['lower']:.4f},{ci['upper']:.4f})**"])
        ca, cb, cc = st.columns(3)
        ca.metric("Lower", f"{ci['lower']:.4f}");
        cb.metric("x-bar", f"{xbc}");
        cc.metric("Upper", f"{ci['upper']:.4f}")
    st.markdown("### 📊 CI Graph")
    fig_ci = draw_confidence_interval_graph(xbc, ci["lower"], ci["upper"], conf)
    st.pyplot(fig_ci)
    buf_ci = io.BytesIO()
    fig_ci.savefig(buf_ci, format='png', dpi=150, bbox_inches='tight')
    buf_ci.seek(0)
    fig_bytes_ci = buf_ci.getvalue()
    show_calculation_download(
        topic="Confidence Interval",
        inputs={"Sample mean (x-bar)": xbc, "Population SD (sigma)": sgc, "Sample size (n)": int(nc),
                "Confidence level": "{}%".format(int(conf * 100))},
        steps=["x-bar = {}, sigma = {}, n = {}, Confidence = {}%".format(xbc, sgc, int(nc), int(conf * 100)),
               "alpha = 1 - {} = {:.2f} -> alpha/2 = {:.4f}".format(conf, 1 - conf, (1 - conf) / 2),
               "z* = norm.ppf(1-alpha/2) = {:.4f}".format(ci['z_star']),
               "SE = sigma/sqrt(n) = {}/sqrt({}) = {:.4f}".format(sgc, int(nc), ci['se']),
               "MoE = z* x SE = {:.4f} x {:.4f} = {:.4f}".format(ci['z_star'], ci['se'], ci['moe']),
               "Lower = x-bar - MoE = {} - {:.4f} = {:.4f}".format(xbc, ci['moe'], ci['lower']),
               "Upper = x-bar + MoE = {} + {:.4f} = {:.4f}".format(xbc, ci['moe'], ci['upper'])],
        results={"z* (critical value)": "{:.4f}".format(ci['z_star']), "Standard Error (SE)": "{:.4f}".format(ci['se']),
                 "Margin of Error (MoE)": "{:.4f}".format(ci['moe']), "Lower Bound": "{:.4f}".format(ci['lower']),
                 "Upper Bound": "{:.4f}".format(ci['upper']),
                 "Confidence Interval": "({:.4f}, {:.4f})".format(ci['lower'], ci['upper'])},
        key_suffix="conf_interval",
        figure_bytes=fig_bytes_ci
    )
    st.markdown("---");
    st.markdown("### ❓ Solve a Text Question")
    q_ci = st.text_area("Enter your question:", key="q_ci")
    if st.button("Solve", key="solve_ci", use_container_width=True):
        solve_text_question(q_ci, "Confidence Interval")
    run_python_code(f"""import math
from scipy.stats import norm
xb={xbc};sg={sgc};n={int(nc)};c={conf}
a=1-c;zs=norm.ppf(1-a/2);SE=sg/math.sqrt(n);MoE=zs*SE
print(f"z*={{zs:.4f}}  SE={{SE:.4f}}  MoE={{MoE:.4f}}")
print("CI=(" + str(round(xb-MoE,4)) + "," + str(round(xb+MoE,4)) + ")")
""", "ci")


# ═══════════════════════════════════════════════════════
#   E F F E C T   O F   S A M P L E   S I Z E
# ═══════════════════════════════════════════════════════
elif menu == "Effect of Sample Size":
    section_title("📈 Effect of Sample Size")
    info_sections(
        "Larger n -> smaller SE and MoE -> narrower CI. Doubling n reduces SE by sqrt(2).",
        "sigma=10. n=25: MoE=3.92. n=100: MoE=1.96. Quadrupling n halves CI width.",
        ["Smaller SE -> more reliable", "Higher statistical power", "Less sensitive to outliers"],
        ["Cost and time increase", "Diminishing returns"],
        ["Clinical Trials", "National Surveys", "ML Datasets", "Opinion Polls"],
        "Poll wants +/-3% MoE at 95%: n=(1.96 x 0.5/0.03)^2 ~ 1068."
    )
    st.markdown("---");
    st.markdown("### 🧮 Manual Solver")
    c1, c2 = st.columns(2)
    with c1:
        sges = st.number_input("sigma", value=10.0, min_value=0.001, key="es_sigma")
        ces = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="es_conf")
    with c2:
        tmoe = st.number_input("Target MoE", value=2.0, min_value=0.01, key="es_moe")
    zses = norm.ppf(1 - (1 - ces) / 2);
    nreq = math.ceil((zses * sges / tmoe) ** 2)
    ca, cb, cc = st.columns(3)
    ca.metric("z*", f"{zses:.4f}");
    cb.metric("Target MoE", f"{tmoe}");
    cc.metric("Min n", f"{nreq}")
    step_block([f"n=(z* x sigma/MoE)^2",
                f"n=({zses:.4f} x {sges}/{tmoe})^2",
                f"n={(zses * sges / tmoe) ** 2:.2f}",
                f"Round up: **n={nreq}**"])
    st.markdown("### 📊 Graphs")
    fig_es = draw_ci_effect(sges)
    st.pyplot(fig_es)
    # Capture figure bytes for PDF
    buf_es = io.BytesIO()
    fig_es.savefig(buf_es, format='png', dpi=150, bbox_inches='tight')
    buf_es.seek(0)
    fig_bytes_es = buf_es.getvalue()
    st.markdown("---");
    st.markdown("### 📋 Effect of Sample Size Table")
    ns_t = [5, 10, 20, 30, 50, 100, 200, 500, 1000]
    se_table = [f"{sges / math.sqrt(n):.4f}" for n in ns_t]
    moe90_table = [f"{norm.ppf(0.95) * sges / math.sqrt(n):.4f}" for n in ns_t]
    moe95_table = [f"{norm.ppf(0.975) * sges / math.sqrt(n):.4f}" for n in ns_t]
    moe99_table = [f"{norm.ppf(0.995) * sges / math.sqrt(n):.4f}" for n in ns_t]
    st.table({"n": ns_t,
              "SE": se_table,
              "MoE (90%)": moe90_table,
              "MoE (95%)": moe95_table,
              "MoE (99%)": moe99_table})
    # Build table data for PDF
    table_data = {
        "Sample Sizes (n)": ns_t,
        "Standard Error (SE)": se_table,
        "MoE (90%)": moe90_table,
        "MoE (95%)": moe95_table,
        "MoE (99%)": moe99_table
    }
    show_calculation_download(
        topic="Effect of Sample Size",
        inputs={"Population SD (sigma)": sges, "Confidence level": "{}%".format(int(ces * 100)), "Target MoE": tmoe},
        steps=["z* = norm.ppf(1-(1-alpha)/2) = {:.4f}".format(zses),
               "n = (z* x sigma / MoE)^2",
               "n = ({:.4f} x {} / {})^2".format(zses, sges, tmoe),
               "n = {:.2f}".format((zses * sges / tmoe) ** 2),
               "Round up: n = {}".format(nreq)],
        results={"z* (critical value)": "{:.4f}".format(zses), "Target MoE": "{}".format(tmoe),
                 "Minimum sample size (n)": "{}".format(nreq), "Formula used": "n = (z* x sigma / MoE)^2",
                 "Sample Size Table": table_data},
        key_suffix="effect_sample_size",
        figure_bytes=fig_bytes_es
    )
    st.markdown("---");
    st.markdown("### ❓ Solve a Text Question")
    q_es = st.text_area("Enter your question:", key="q_es")
    if st.button("Solve", key="solve_es", use_container_width=True):
        solve_text_question(q_es, "Effect of Sample Size")
    run_python_code(f"""import math
from scipy.stats import norm
sg={sges};c={ces};zs=norm.ppf(1-(1-c)/2)
print(f"z*={{zs:.4f}}")
for n in [5,10,20,30,50,100,200,500,1000]:
    se=sg/math.sqrt(n);moe=zs*se
    print(f"  n={{n:5d}}  SE={{se:.4f}}  MoE={{moe:.4f}}")
print(f"\\nFor MoE={tmoe}: n>="+str(math.ceil((zs*sg/{tmoe})**2)))
""", "effectsample")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
show_footer()
st.sidebar.markdown("---")
