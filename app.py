import re

import numpy as np
import streamlit as st
from PIL import Image

import inference

st.set_page_config(
    page_title="PneumoVision · Explainable Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1180px; }
.stApp { background: linear-gradient(180deg,#EEF4FB 0%,#F7FAFD 320px,#F7FAFD 100%); }

/* top bar */
.topbar { display:flex; align-items:center; justify-content:space-between; padding:6px 4px 18px; }
.brand { display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.15rem; color:#0B2545; }
.logo { width:36px; height:36px; border-radius:10px; display:grid; place-items:center; font-size:1.2rem;
        background:linear-gradient(135deg,#0F6CBD,#35A7E8); color:#fff; }
.pill { font-size:.75rem; font-weight:600; color:#0F6CBD; background:#E3F0FC; padding:5px 12px; border-radius:999px; }

/* hero */
.hero { background:linear-gradient(120deg,#0B2545 0%,#13477F 55%,#1C7BC0 100%); color:#fff; border-radius:24px;
        padding:44px 48px; position:relative; overflow:hidden; box-shadow:0 18px 40px rgba(11,37,69,.18); }
.hero:after { content:""; position:absolute; right:-80px; top:-80px; width:340px; height:340px; border-radius:50%;
              background:radial-gradient(circle,rgba(255,255,255,.18),transparent 70%); }
.hero h1 { font-size:2.4rem; line-height:1.15; font-weight:800; margin:0 0 12px; color:#fff; }
.hero p { font-size:1.05rem; color:#CFE3F6; max-width:640px; margin:0 0 20px; }
.chips { display:flex; flex-wrap:wrap; gap:10px; }
.chip { background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22); padding:6px 14px; border-radius:999px;
        font-size:.82rem; font-weight:500; color:#fff; }

/* steps */
.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:22px 0 8px; }
.step { background:#fff; border:1px solid #E1E9F2; border-radius:16px; padding:16px 18px; display:flex; gap:14px; align-items:center; }
.step .n { width:34px; height:34px; border-radius:50%; background:#E3F0FC; color:#0F6CBD; font-weight:800; display:grid; place-items:center; flex:none; }
.step b { color:#0B2545; display:block; font-size:.95rem; }
.step span { color:#5B6B7F; font-size:.82rem; }

/* cards */
.card { background:#fff; border:1px solid #E1E9F2; border-radius:20px; padding:24px 26px; box-shadow:0 6px 20px rgba(16,35,58,.05); }
.card h3 { margin:0 0 4px; font-size:1.05rem; color:#0B2545; font-weight:700; }
.muted { color:#5B6B7F; font-size:.85rem; }
.sec-title { font-size:1.35rem; font-weight:800; color:#0B2545; margin:30px 0 14px; }

/* uploader */
[data-testid="stFileUploader"] section { background:#F4F9FE; border:2px dashed #9CC6EC; border-radius:16px; padding:28px; }
[data-testid="stFileUploader"] section:hover { border-color:#0F6CBD; background:#EAF3FC; }
[data-testid="stFileUploader"] button { border-radius:10px; }

/* verdict */
.verdict { border-radius:22px; padding:26px 30px; display:flex; align-items:center; gap:26px; border:1px solid; }
.verdict.bad { background:linear-gradient(120deg,#FFF1F0,#FFE4E1); border-color:#F5B5AE; }
.verdict.good { background:linear-gradient(120deg,#EDFBF4,#D9F5E8); border-color:#9ADBBB; }
.verdict.warn { background:linear-gradient(120deg,#FFF8E6,#FEEFC7); border-color:#F2CE7A; }
.ring { --p:0; --c:#0F6CBD; width:112px; height:112px; border-radius:50%; flex:none; display:grid; place-items:center;
        background:conic-gradient(var(--c) calc(var(--p)*1%), rgba(16,35,58,.10) 0); }
.ring div { width:86px; height:86px; border-radius:50%; background:#fff; display:grid; place-items:center;
            font-weight:800; font-size:1.2rem; color:#0B2545; }
.verdict .tag { font-size:.72rem; letter-spacing:.08em; font-weight:700; text-transform:uppercase; opacity:.75; }
.verdict h2 { margin:2px 0 6px; font-size:1.9rem; font-weight:800; }
.verdict.bad h2 { color:#B42318; } .verdict.good h2 { color:#067647; } .verdict.warn h2 { color:#B54708; }
.verdict p { margin:0; color:#33475B; font-size:.95rem; }

/* metric tiles */
.tiles { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:16px; }
.tile { background:#fff; border:1px solid #E1E9F2; border-radius:16px; padding:16px 18px; }
.tile small { color:#5B6B7F; font-weight:600; font-size:.74rem; text-transform:uppercase; letter-spacing:.05em; }
.tile div { font-size:1.35rem; font-weight:800; color:#0B2545; margin-top:2px; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:6px; background:#E9F0F8; padding:5px; border-radius:14px; width:fit-content; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:8px 18px; font-weight:600; height:auto; }
.stTabs [aria-selected="true"] { background:#fff; color:#0F6CBD; box-shadow:0 2px 8px rgba(16,35,58,.1); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

/* report blocks */
.rb h4, .rb p { margin-top:0 !important; margin-bottom:4px !important; padding:0 !important; }
.rb ul { margin:2px 0 0 !important; }
.rb { background:#fff; border:1px solid #E1E9F2; border-left:5px solid #0F6CBD; border-radius:14px; padding:13px 20px; margin-bottom:10px; }
.rb h4 { margin:0 0 6px; font-size:.98rem; color:#0B2545; font-weight:700; }
.rb p { margin:0 0 6px; } .rb p, .rb li { color:#33475B; font-size:.93rem; line-height:1.55; }
.rb ul { margin:0; padding-left:20px; }
.rb.urgent { border-left-color:#D92D20; background:#FFF8F7; }
.rb.doctor { border-left-color:#067647; background:#F4FCF8; }
.rb.prec { border-left-color:#B54708; }
.rb.sug { border-left-color:#7A5AF8; }

.disclaimer { background:#0B2545; color:#CFE3F6; border-radius:16px; padding:16px 22px; font-size:.84rem; margin-top:30px; }
.disclaimer b { color:#fff; }
.footer { text-align:center; color:#8696A8; font-size:.78rem; margin-top:22px; }
[data-testid="stImage"] img { border-radius:14px; }
@media (max-width:760px){ .steps,.tiles{grid-template-columns:1fr;} .hero{padding:28px 24px;} .hero h1{font-size:1.7rem;}
  .verdict{flex-direction:column; text-align:center;} }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
def looks_like_xray(img):
    """Reject obvious colour photos (X-rays are grayscale)."""
    a = np.asarray(img.convert("RGB")).astype(int)
    diff = (abs(a[..., 0] - a[..., 1]) + abs(a[..., 1] - a[..., 2])) / 2
    return diff.mean() < 12


@st.cache_resource(show_spinner="Loading the Vision Transformer…")
def get_model():
    return inference.load_model()


def md_inline(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)


def report_blocks(md):
    """Turn the report text into styled cards.
    A line that starts with **Label** (optionally after "- ") opens a new card;
    "- " lines are bullets and other lines are paragraphs of the current card."""
    cards = []  # each: [title, paragraphs, bullets]
    for raw in md.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        body = line[2:].strip() if line[:2] in ("- ", "* ", "• ") else line
        is_bullet = body is not line
        m = re.match(r"^\*\*(.+?)\*\*:?\s*(.*)$", body)
        if m:
            title = m.group(1).strip().rstrip(":.").strip()
            rest = m.group(2).strip()
            cards.append([title, [rest] if rest else [], []])
        elif cards and is_bullet:
            cards[-1][2].append(body)
        elif cards:
            cards[-1][1].append(body)
        else:
            cards.append(["Result", [body], []])
    html = []
    for title, paras, bullets in cards:
        inner = "".join(f"<p>{md_inline(p)}</p>" for p in paras)
        if bullets:
            inner += "<ul>" + "".join(f"<li>{md_inline(x)}</li>" for x in bullets) + "</ul>"
        t = title.lower()
        cls = ("urgent" if "urgent" in t else
               "doctor" if "consult" in t or "next steps" in t else
               "prec" if "precaution" in t else
               "sug" if "suggestion" in t else "")
        html.append(f'<div class="rb {cls}"><h4>{title}</h4>{inner}</div>')
    return "".join(html)


# ---------------------------------------------------------------- top bar + hero
st.markdown(
    """
<div class="topbar">
  <div class="brand"><div class="logo">🫁</div>PneumoVision</div>
  <div class="pill">Research prototype · ViT-B/16</div>
</div>
<div class="hero">
  <h1>See what the AI sees,<br>in plain language.</h1>
  <p>Upload a chest X-ray. A Vision Transformer screens it for pneumonia, highlights the lung regions that
     drove its decision, and explains the result in words anyone can follow.</p>
  <div class="chips">
    <span class="chip">🧠 Vision Transformer</span><span class="chip">🔥 Attention heatmap</span>
    <span class="chip">📝 Plain-language report</span><span class="chip">🎯 Calibrated confidence</span>
  </div>
</div>
<div class="steps">
  <div class="step"><div class="n">1</div><div><b>Upload</b><span>JPG or PNG chest X-ray</span></div></div>
  <div class="step"><div class="n">2</div><div><b>Analyse</b><span>Model screens and maps attention</span></div></div>
  <div class="step"><div class="n">3</div><div><b>Understand</b><span>Heatmap and easy-to-read report</span></div></div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- upload
st.markdown('<div class="sec-title">Upload an X-ray</div>', unsafe_allow_html=True)
up_col, tip_col = st.columns([2, 1], gap="large")
with up_col:
    file = st.file_uploader("Chest X-ray image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
with tip_col:
    st.markdown(
        """<div class="card"><h3>For best results</h3>
        <p class="muted">• Frontal (PA/AP) chest view<br>• Grayscale X-ray, not a phone photo of a screen<br>
        • Do <b>not</b> upload real patient data</p></div>""",
        unsafe_allow_html=True,
    )

if file is None:
    st.markdown(
        '<div class="footer">Waiting for an image… your result will appear here.</div>',
        unsafe_allow_html=True,
    )
else:
    try:
        img = Image.open(file).convert("RGB")
    except Exception:
        st.error("That file could not be opened as an image. Please upload a JPG or PNG.")
        st.stop()

    if not looks_like_xray(img):
        st.markdown(
            """<div class="verdict warn"><div><div class="tag">Image check</div>
            <h2>This doesn't look like an X-ray</h2>
            <p>The image is in colour. Chest X-rays are grayscale, so please upload a proper X-ray image.</p></div></div>""",
            unsafe_allow_html=True,
        )
        st.stop()

    with st.spinner("Analysing the X-ray…"):
        result = inference.analyze(get_model(), img)

    label, conf = result["label"], float(result["confidence"])
    uncertain = conf < 0.90
    if uncertain:
        cls, ring_c, tag, head = "warn", "#F79009", "Inconclusive", "Result uncertain"
        sub = "The model isn't confident enough to call this. Please have a doctor review the scan."
    elif label == "PNEUMONIA":
        cls, ring_c, tag, head = "bad", "#D92D20", "Model finding", "Signs of pneumonia"
        sub = "The scan shows patterns the model associates with pneumonia. See a doctor for confirmation."
    else:
        cls, ring_c, tag, head = "good", "#12B76A", "Model finding", "No signs of pneumonia"
        sub = "The scan looks normal to the model. This does not replace a medical opinion."
    level = "Low" if uncertain else ("High" if conf >= 0.97 else "Moderate")

    st.markdown('<div class="sec-title">Your result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="verdict {cls}">
          <div class="ring" style="--p:{conf*100:.1f};--c:{ring_c}"><div>{conf:.1%}</div></div>
          <div><div class="tag">{tag}</div><h2>{head}</h2><p>{sub}</p></div>
        </div>
        <div class="tiles">
          <div class="tile"><small>Prediction</small><div>{label.title()}</div></div>
          <div class="tile"><small>Confidence</small><div>{conf:.2%}</div></div>
          <div class="tile"><small>Confidence level</small><div>{level}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    tab_map, tab_report, tab_about = st.tabs(["🔥 Heatmap", "📝 Plain-language report", "ℹ️ How it works"])

    with tab_map:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<h3 style="margin:0;color:#0B2545;font-size:1.05rem">Original X-ray</h3><p class="muted">As uploaded</p>', unsafe_allow_html=True)
            st.image(img, width="stretch")
        with c2:
            st.markdown('<h3 style="margin:0;color:#0B2545;font-size:1.05rem">Where the model looked</h3><p class="muted">Warmer colours = more attention</p>', unsafe_allow_html=True)
            st.image(result["overlay"], width="stretch")
        st.caption("Attention shows where the model focused, not a confirmed location of infection.")

    with tab_report:
        st.markdown(report_blocks(result["report"]), unsafe_allow_html=True)

    with tab_about:
        st.markdown(
            """<div class="card"><h3>What happens behind the scenes</h3>
            <p class="muted" style="font-size:.93rem;line-height:1.65">
            <b>1. Patches:</b> the X-ray is cut into 16×16 pixel patches.<br>
            <b>2. Attention:</b> a Vision Transformer (ViT-B/16) lets every patch look at every other patch across 12 layers.<br>
            <b>3. Heatmap:</b> attention rollout combines those layers to show which lung regions mattered most.<br>
            <b>4. Calibration:</b> temperature scaling keeps the confidence honest; anything under 90% is marked uncertain.<br>
            <b>5. Report:</b> the result and attention zones are turned into a plain-language explanation.</p></div>""",
            unsafe_allow_html=True,
        )

st.markdown(
    """<div class="disclaimer"><b>Important:</b> This is a research prototype for educational use, not a medical device.
    It cannot diagnose any condition. Always consult a qualified doctor about your health.</div>
    <div class="footer">PneumoVision · Attention-based explainable pneumonia detection · ViT-B/16 vs ResNet50</div>""",
    unsafe_allow_html=True,
)