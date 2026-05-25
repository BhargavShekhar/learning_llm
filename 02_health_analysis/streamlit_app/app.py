import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import io
import os

# Load environment variables
load_dotenv()


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF using pypdf."""
    try:
        import pypdf
    except ImportError:
        st.error("pypdf is not installed. Run `pip install pypdf` and restart.")
        st.stop()

    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip() 
    if not text:
        st.error(
            "Could not extract text from this PDF. It may be a scanned image — "
            "please paste the report text manually."
        )
        st.stop()
    return text


def llm_text(response) -> str:
    """
    Safely pull a plain string out of a LangChain response.

    .content can be:
      - a plain str  (older LangChain / some models)
      - a list of dicts  [{"type": "text", "text": "…"}, …]
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile"
    )


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Blood Work Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ---- hero header ---- */
    .hero {
        background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 60%, #5ba4cf 100%);
        border-radius: 16px;
        padding: 2.5rem 2.8rem;
        color: white;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "🩺";
        font-size: 7rem;
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        opacity: .15;
    }
    .hero h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        margin: 0 0 .4rem 0;
        line-height: 1.2;
    }
    .hero p {
        margin: 0;
        opacity: .85;
        font-size: .97rem;
        font-weight: 300;
    }

    /* ---- metric cards ---- */
    .metric-card {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: .8rem;
        border-left: 5px solid;
        background: #f8fafc;
    }
    .metric-HIGH  { border-color: #e74c3c; background: #fff5f5; }
    .metric-LOW   { border-color: #f39c12; background: #fffbf0; }
    .metric-NORMAL{ border-color: #27ae60; background: #f0fff4; }

    .metric-card .badge {
        display: inline-block;
        font-size: .7rem;
        font-weight: 600;
        padding: .15rem .55rem;
        border-radius: 20px;
        letter-spacing: .05em;
        margin-left: .5rem;
    }
    .badge-HIGH   { background: #fde8e8; color: #c0392b; }
    .badge-LOW    { background: #fef3cd; color: #9a6000; }
    .badge-NORMAL { background: #d4edda; color: #155724; }

    /* ---- section headings ---- */
    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.25rem;
        color: #0f4c75;
        border-bottom: 2px solid #e2eaf2;
        padding-bottom: .4rem;
        margin-bottom: 1rem;
    }

    /* ---- disclaimer box ---- */
    .disclaimer {
        background: #eaf4fb;
        border-radius: 10px;
        padding: .8rem 1.2rem;
        font-size: .82rem;
        color: #2c6e9b;
        border: 1px solid #b8d9ef;
        margin-top: 1rem;
    }

    /* ---- sample code block ---- */
    .stCodeBlock { border-radius: 10px !important; }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #f0f7ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── hero ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <h1>Blood Work Analysis</h1>
        <p>Upload or paste your blood report — get instant AI-powered insights
        and personalised Indian dietary guidance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🩺 How it works")
    st.markdown(
        """
        1. **Upload** a PDF / TXT file **or paste** your report text  
        2. Click **Analyse**  
        3. Review extracted values, health summary, and your personalised diet plan  
        4. Download the full report
        """
    )
    st.divider()
    st.markdown("### ⚙️ Settings")
    show_raw = st.toggle("Show raw LLM output", value=False)
    st.divider()
    st.caption("Powered by Gemini · LangChain · Streamlit")

# ── input area ────────────────────────────────────────────────────────────────

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<p class="section-title">📋 Your Blood Report</p>', unsafe_allow_html=True)

    method = st.radio("Input method", ["📎 Upload File", "📝 Paste Text"], horizontal=True, label_visibility="collapsed")

    blood_report_text: str | None = None

    if method == "📎 Upload File":
        uploaded = st.file_uploader(
            "Upload TXT or PDF", type=["txt", "pdf", "md"],
            help="PDF text extraction works for digital (non-scanned) reports."
        )
        if uploaded:
            raw_bytes = uploaded.read()
            if uploaded.name.lower().endswith(".pdf"):
                with st.spinner("Extracting text from PDF…"):
                    blood_report_text = extract_text_from_pdf(raw_bytes)
            else:
                blood_report_text = raw_bytes.decode("utf-8", errors="replace")
            st.success(f"✅ **{uploaded.name}** loaded — {len(blood_report_text):,} characters")
    else:
        blood_report_text = st.text_area(
            "Paste report",
            height=280,
            placeholder=(
                "Hemoglobin: 15.1 g/dL (Normal: 13.5–17.5)\n"
                "Total Cholesterol: 238 mg/dL (Normal: <200)\n…"
            ),
            label_visibility="collapsed",
        )

with right:
    st.markdown('<p class="section-title">📄 Sample Report</p>', unsafe_allow_html=True)
    st.code(
        """\
Patient: John Doe, Age 45, Male
Date: May 25, 2026

COMPLETE BLOOD COUNT
Hemoglobin: 15.1 g/dL (Normal: 13.5-17.5)
WBC: 6.8 x10^3/uL (Normal: 4.5-11.0)

LIPID PANEL
Total Cholesterol: 238 mg/dL (Normal: <200)
LDL: 162 mg/dL (Normal: <100)
HDL: 36 mg/dL (Normal: >40)
Triglycerides: 188 mg/dL (Normal: <150)

METABOLIC PANEL
Glucose: 92 mg/dL (Normal: 70-99)
HbA1c: 5.3% (Normal: <5.7%)
Creatinine: 1.0 mg/dL (Normal: 0.7-1.3)""",
        language="text",
    )

# ── analyse button ────────────────────────────────────────────────────────────

st.write("")
analyse = st.button("🔍 Analyse Blood Work", type="primary", use_container_width=True)

# ── analysis ──────────────────────────────────────────────────────────────────

if analyse:
    if not blood_report_text or not blood_report_text.strip():
        st.error("❌ Please provide blood work data before analysing.")
        st.stop()

    llm = get_llm()

    # Stage 1 — extract + classify values
    extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL
based on the reference ranges provided in the report.

Format EVERY line exactly as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference Range: range

Blood Report:
{blood_report_text}
"""
    with st.spinner("Stage 1 of 2 — Extracting and classifying values…"):
        extracted_values = llm_text(llm.invoke(extraction_prompt))

    # Stage 2 — health summary + diet
    diet_prompt = f"""
You are a clinical nutritionist specialising in Indian dietary habits.

Based on the blood work analysis below, provide:
1. A short health summary (4–5 sentences) in simple, patient-friendly language.
2. A practical Indian diet plan with exactly two sections:
   ## Foods to Avoid
   ## Foods to Eat More Of

Do not add any other sections.

Blood Work Analysis:
{extracted_values}
"""
    with st.spinner("Stage 2 of 2 — Generating health summary and diet plan…"):
        diet_plan = llm_text(llm.invoke(diet_prompt))

    st.success("✅ Analysis complete!")
    st.divider()

    # ── parse extracted values into cards ────────────────────────────────────

    def parse_lines(raw: str):
        entries = []
        for line in raw.splitlines():
            line = line.strip().lstrip("-").strip()
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            name_val = parts[0]
            status = ""
            ref = ""
            for p in parts[1:]:
                if p.upper().startswith("STATUS"):
                    status = p.split(":", 1)[-1].strip().upper()
                elif p.upper().startswith("REFERENCE"):
                    ref = p.split(":", 1)[-1].strip()
            entries.append({"label": name_val, "status": status, "ref": ref})
        return entries

    entries = parse_lines(extracted_values)

    # Counts
    high_count   = sum(1 for e in entries if e["status"] == "HIGH")
    low_count    = sum(1 for e in entries if e["status"] == "LOW")
    normal_count = sum(1 for e in entries if e["status"] == "NORMAL")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tests analysed", len(entries))
    m2.metric("🔴 High",  high_count)
    m3.metric("🟡 Low",   low_count)
    m4.metric("🟢 Normal", normal_count)

    st.write("")
    tab1, tab2, tab3 = st.tabs(["📊 Extracted Values", "📈 Health Summary", "🍽️ Diet Plan"])

    with tab1:
        st.markdown('<p class="section-title">Classified Blood Test Values</p>', unsafe_allow_html=True)
        if entries:
            for e in entries:
                status = e["status"] if e["status"] in ("HIGH", "LOW", "NORMAL") else "NORMAL"
                badge_cls = f"badge-{status}"
                card_cls  = f"metric-{status}"
                ref_html  = f" <small style='color:#888;'>({e['ref']})</small>" if e["ref"] else ""
                st.markdown(
                    f"""<div class="metric-card {card_cls}">
                        {e['label']}{ref_html}
                        <span class="badge {badge_cls}">{status}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            # fallback — just render raw markdown
            st.markdown(extracted_values)

        if show_raw:
            with st.expander("Raw LLM output"):
                st.text(extracted_values)

    with tab2:
        st.markdown('<p class="section-title">Your Health Summary</p>', unsafe_allow_html=True)
        # Everything before the first "##" heading (the diet section)
        if "##" in diet_plan:
            summary = diet_plan.split("##")[0].strip()
        elif "Foods to" in diet_plan:
            summary = diet_plan.split("Foods to")[0].strip()
        else:
            summary = diet_plan

        # Strip leading "1." if present
        summary = summary.lstrip("1.").strip()
        st.markdown(summary)

    with tab3:
        st.markdown('<p class="section-title">Personalised Indian Diet Plan</p>', unsafe_allow_html=True)
        # Show only the diet part (from first ## onwards)
        if "##" in diet_plan:
            diet_section = "##" + "##".join(diet_plan.split("##")[1:])
        elif "Foods to" in diet_plan:
            idx = diet_plan.index("Foods to")
            diet_section = diet_plan[idx:]
        else:
            diet_section = diet_plan
        st.markdown(diet_section)

    # ── download ──────────────────────────────────────────────────────────────
    st.divider()
    full_report = f"""BLOOD WORK ANALYSIS REPORT
Generated by Blood Work Analysis AI
=====================================

EXTRACTED & CLASSIFIED VALUES
------------------------------
{extracted_values}

HEALTH SUMMARY & DIET PLAN
---------------------------
{diet_plan}
"""
    dl_col, note_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            label="📥 Download Full Report (TXT)",
            data=full_report,
            file_name="blood_work_analysis.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with note_col:
        st.markdown(
            '<div class="disclaimer">💡 <b>Note:</b> This analysis is for informational purposes only. '
            "Always consult a qualified healthcare professional for diagnosis and treatment.</div>",
            unsafe_allow_html=True,
        )

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:.78rem;'>"
    "Blood Work Analysis AI · Not a substitute for professional medical advice"
    "</p>",
    unsafe_allow_html=True,
)