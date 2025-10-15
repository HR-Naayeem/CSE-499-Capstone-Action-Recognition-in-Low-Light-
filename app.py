import streamlit as st
import numpy as np
import pandas as pd
import cv2
import torch
import joblib
from torchvision import models, transforms
from PIL import Image
import io, zipfile, json, tempfile, os, shutil, time
import matplotlib.pyplot as plt
import base64, pathlib

# ---------------------------
# Base64 background images (with safe fallback)
# ---------------------------
TRANSPARENT_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)
def _b64_image(path: str) -> str:
    try:
        p = pathlib.Path(path)
        if p.is_file():
            return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception:
        pass
    return TRANSPARENT_PNG_B64

BG_MAIN_B64 = _b64_image("assets/bg_main.png")
BG_SIDEBAR_B64 = _b64_image("assets/bg_sidebar.png")
if BG_SIDEBAR_B64 == TRANSPARENT_PNG_B64 and BG_MAIN_B64 != TRANSPARENT_PNG_B64:
    BG_SIDEBAR_B64 = BG_MAIN_B64

# ---------------------------
# Page + CSS (aurora + vignette + grain + image backgrounds + SIDEBAR)
# ---------------------------
full_css = """
<style>
:root{
  /* page palette + boosted aurora glows */
  --bg1:#050a14; --bg2:#0a1020; --bg3:#0b1220;
  --glow1:rgba(127,90,240,.35);
  --glow2:rgba(0,200,150,.28);
  --glow3:rgba(61,157,243,.32);
}

/* ===== MAIN CANVAS (BG image under aurora) ===== */
[data-testid="stAppViewContainer"]{
  position: relative;
  background:
    linear-gradient(rgba(5,10,20,.55), rgba(5,10,20,.55)),
    url("data:image/png;base64,__BG_MAIN__") center/cover fixed no-repeat;
}
[data-testid="stAppViewContainer"]::before{
  content:""; position: fixed; inset:-20%;
  background:
    radial-gradient(600px 320px at 20% 8%,   var(--glow1), transparent 60%),
    radial-gradient(700px 360px at 82% 24%,  var(--glow2), transparent 65%),
    radial-gradient(800px 420px at 50% 90%,  var(--glow3), transparent 70%);
  /* stronger, crisper aurora */
  filter: blur(30px) saturate(160%) brightness(1.12);
  opacity: 0.95;
  animation: aurora 22s ease-in-out infinite alternate;
  pointer-events:none; z-index:0;
}
[data-testid="stAppViewContainer"]::after{
  content:""; position: fixed; inset:0;
  background:
    radial-gradient(1200px 600px at 50% 10%, rgba(0,0,0,.25), transparent 70%),
    radial-gradient(1400px 900px at 50% 100%, rgba(0,0,0,.35), transparent 75%),
    radial-gradient(100px 100px at 0 0, rgba(255,255,255,.02) 1px, transparent 1px);
  background-size:auto,auto,3px 3px;
  mix-blend-mode: overlay; pointer-events:none; z-index:0;
}

/* ===== WIDE CONTENT (no shrink) ===== */
.block-container, section.main > div.block-container{
  max-width: 1400px !important;
  padding-left: 2rem; padding-right: 2rem;
  position:relative; z-index:1;
}

/* ===== SIDEBAR (BG image + aurora only; default widgets) ===== */
[data-testid="stSidebar"]{
  position: relative;
  background:
    linear-gradient(rgba(5,10,20,.60), rgba(5,10,20,.60)),
    url("data:image/png;base64,__BG_SIDEBAR__") center/cover no-repeat;
  overflow: hidden;
}
[data-testid="stSidebar"]::before{
  content:""; position: absolute; inset:-12%;
  background:
    radial-gradient(260px 160px at 20% 8%,   var(--glow1), transparent 60%),
    radial-gradient(300px 200px at 84% 26%,  var(--glow2), transparent 65%),
    radial-gradient(320px 220px at 50% 95%,  var(--glow3), transparent 70%);
  filter: blur(26px) saturate(170%) brightness(1.12);
  opacity: 0.95;
  animation: aurora 18s ease-in-out infinite alternate;
  pointer-events:none; z-index:0;
}
[data-testid="stSidebar"]::after{
  content:""; position: absolute; inset:0;
  background:
    radial-gradient(500px 200px at 50% 0%,   rgba(0,0,0,.25), transparent 70%),
    radial-gradient(700px 380px at 50% 100%, rgba(0,0,0,.35), transparent 80%),
    radial-gradient(100px 100px at 0 0, rgba(255,255,255,.02) 1px, transparent 1px);
  background-size:auto,auto,3px 3px;
  mix-blend-mode: overlay; pointer-events:none; z-index:0;
}
[data-testid="stSidebar"] > * { position: relative; z-index: 1; }

/* ===== Components you wanted smaller/narrower ===== */

/* video preview a little smaller & centered */
.stVideo{ max-width: 600px; margin: .25rem auto .75rem auto; }
.stVideo video{ width:100% !important; border-radius:12px; }

/* narrow / centered wrapper for Top-K table & chart */
.pred-wrap{ max-width: 560px; margin: 0 auto; }

/* center any images (bar chart) that appear inside pred-wrap */
.pred-wrap .stImage, .pred-wrap img{ display:block; margin:0 auto; }

/* clamp st.dataframe width when placed inside pred-wrap */
.pred-wrap [data-testid="stDataFrame"]{
  max-width: 560px !important;
  margin: 0 auto !important;
}
.pred-wrap [data-testid="stDataFrame"] > div{ width:100% !important; }
.pred-wrap [data-testid="stDataFrame"] [role="grid"]{ max-width:560px !important; }

/* ===== helpers ===== */
@keyframes aurora{
  0%{   transform: translateY(-2%) rotate(0deg)   scale(1.02); }
  100%{ transform: translateY( 2%) rotate(180deg) scale(1.02); }
}
</style>
"""
full_css = full_css.replace("__BG_MAIN__", BG_MAIN_B64).replace("__BG_SIDEBAR__", BG_SIDEBAR_B64)
st.markdown(full_css, unsafe_allow_html=True)


# ---------------------------
# Model + labels (original pipeline)
# ---------------------------
clf = joblib.load("rf_model.pkl")
label_to_idx = joblib.load("label_to_idx.pkl")
idx_to_label = {v: k for k, v in label_to_idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# Helpers (temp I/O, plot, zip)
# ---------------------------
def save_upload_to_temp(uploaded):
    tmp_dir = tempfile.mkdtemp(prefix="vid_")
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    path = os.path.join(tmp_dir, f"upload{suffix}")
    uploaded.seek(0)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path, tmp_dir

def cleanup_temp(path, tmp_dir):
    for _ in range(8):
        try:
            if os.path.exists(path): os.remove(path)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            break
        except PermissionError:
            time.sleep(0.25)

def extract_features_from_video(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None, []

    feats = []
    with torch.no_grad():
        for frame in frames:
            x = transform(frame).unsqueeze(0).to(device)
            feat = resnet(x).squeeze().cpu().numpy()
            feats.append(feat)
    return np.mean(np.stack(feats, axis=0), axis=0).reshape(1, -1), frames

def bar_png(labels, scores_pct):
    # smaller figure + smaller fonts + tighter layout
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, scores_pct[::-1])
    ax.set_yticks(y, labels[::-1])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=10, labelpad=4)
    ax.tick_params(axis='both', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def make_zip(preview_imgs_rgb, chart_png, df_csv, report_json):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("topk.csv", df_csv)
        z.writestr("topk_chart.png", chart_png or b"")
        z.writestr("report.json", report_json)
        for i, img in enumerate(preview_imgs_rgb, 1):
            pil = Image.fromarray(img)
            b = io.BytesIO(); pil.save(b, format="PNG")
            z.writestr(f"preview_frame_{i:02d}.png", b.getvalue())
    mem.seek(0)
    return mem

# ---------------------------
# Sidebar (wrapped in a card)
# ---------------------------
with st.sidebar:
    st.markdown('<div class="sbcard">', unsafe_allow_html=True)
    st.markdown("### Controls")
    max_frames = st.slider("Frames sampled per video", 8, 48, 16)
    preview_n = st.slider("Preview frames to show", 1, 3, 3)
    top_k = st.slider("Top-K predictions", 3, 5, 5, step=1)
    show_table = st.checkbox("Show Top-K predictions", value=True)
    show_bar = st.checkbox("Show Top-K bar chart", value=True)
    show_video = st.checkbox("Show video preview", value=True)
    show_frames = st.checkbox("Show preview frames", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Main UI
# ---------------------------
st.title("🎬 Action Recognition in Low Light")

uploader = st.file_uploader(
    "Upload a .mp4 or .avi video",
    type=["mp4", "avi", "mpeg4", "mov"],
    accept_multiple_files=False
)

if uploader is None:
    st.info("Upload a video to get started.")
else:
    if show_video:
        st.video(uploader)  # CSS makes it narrower

    temp_path, temp_dir = save_upload_to_temp(uploader)
    try:
        with st.spinner("Analyzing video..."):
            features, all_frames = extract_features_from_video(temp_path, max_frames=max_frames)
    finally:
        cleanup_temp(temp_path, temp_dir)

    if features is None:
        st.error("⚠️ Could not read frames from video.")
        st.stop()

    # Predict (Top-K)
    probs = clf.predict_proba(features)[0]
    topk_idx = np.argsort(probs)[::-1][:top_k]
    topk_labels = [idx_to_label[i] for i in topk_idx]
    topk_scores_pct = (probs[topk_idx] * 100).round(2)

    # Top-1 summary
    action_sentences = {
        "Sit": "sitting", "Stand": "standing", "Run": "running", "Jump": "jumping",
        "Walk": "walking", "Drink": "drinking", "Pick": "picking something up",
        "Push": "pushing", "Pour": "pouring", "Wave": "waving", "Turn": "turning"
    }
    top1 = topk_labels[0]
    st.markdown('<div class="narrow">', unsafe_allow_html=True)
    st.success(f"✅ Predicted Action: The person is {action_sentences.get(top1, top1.lower()+'ing')}.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Centered + narrower TABLE & CHART ----------
    # Prepare df once (also used for downloads later)
    df = pd.DataFrame({"Label": topk_labels, "Confidence (%)": topk_scores_pct})

    # 3 columns: left spacer, narrow center, right spacer
    # Adjust the middle weight to change width (1.10 = narrow, 1.30 = wider)
    left, mid, right = st.columns([1, 2, 1], gap="large")

    with mid:
        if show_table:
            st.subheader(f"🔝 Top-{top_k} Predictions")
            st.dataframe(df, use_container_width=True, hide_index=True, height=220)
        else:
            st.caption("Enable “Show Top-K predictions” from the sidebar to display the table here.")

        st.subheader("Top-K Bar Chart")
        chart_png_bytes = bar_png(topk_labels, topk_scores_pct.tolist()) if show_bar else None
        if show_bar and chart_png_bytes:
            # Fit to column width → perfectly centered & narrower
            st.image(chart_png_bytes, caption="Top-K confidence", use_container_width=True)
        else:
            st.caption("Enable the bar chart from the sidebar to display it here.")
    # ---------- /TABLE & CHART ----------

    # Downloads (stable keys; no auto-downloads)
    st.markdown('<div class="narrow">', unsafe_allow_html=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    report = {
        "file": uploader.name,
        "frames_sampled": int(max_frames),
        "preview_frames": int(preview_n),
        "k": int(top_k),
        "topk": [{"label": l, "confidence_percent": float(s)} for l, s in zip(topk_labels, topk_scores_pct)],
    }
    json_bytes = json.dumps(report, indent=2).encode("utf-8")

    st.download_button("Download Top-K CSV", csv_bytes,
                       file_name=f"top{top_k}.csv", mime="text/csv", key="dl_csv")
    st.download_button("Download Report (JSON)", json_bytes,
                       file_name="report.json", mime="application/json", key="dl_json")
    st.download_button("Download Chart (PNG)", chart_png_bytes or b"",
                       file_name=f"top{top_k}_chart.png", mime="image/png",
                       key="dl_chart", disabled=not show_bar)

    zip_bytes = make_zip(all_frames[:preview_n], chart_png_bytes, csv_bytes, json_bytes)
    st.download_button("Download All (ZIP)", zip_bytes,
                       file_name="inference_outputs.zip", mime="application/zip", key="dl_zip")
    st.markdown('</div>', unsafe_allow_html=True)

    # Preview frames
    if show_frames:
        st.markdown('<div class="center">', unsafe_allow_html=True)
        st.subheader("🎞️ Preview Frames")
        cols = st.columns(3, gap="large")
        previews_rgb = all_frames[:preview_n]
        while len(previews_rgb) < 3:
            previews_rgb.append(previews_rgb[-1] if previews_rgb else np.zeros((224,224,3), dtype=np.uint8))
        for c, img in zip(cols, previews_rgb[:3]):
            c.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

