import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, sosfiltfilt, welch, spectrogram, iirnotch, filtfilt
from scipy.integrate import trapezoid
from sklearn.ensemble import IsolationForest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(page_title="NeuroScan AI | EEG Analyzer", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#091525 0%,#10263b 55%,#123044 100%); color:#f4f7fb}
section[data-testid="stSidebar"] {background:#101e30}
div[data-testid="stMetric"] {background:#172b40;padding:14px;border-radius:12px;border:1px solid #31506b}
h1,h2,h3 {color:#f4f7fb}
.small-muted {color:#bdcbd9}
</style>
""", unsafe_allow_html=True)

BANDS = {
    "Delta (0.5–4 Hz)": (0.5, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–13 Hz)": (8, 13),
    "Beta (13–30 Hz)": (13, 30),
    "Gamma (30–45 Hz)": (30, 45),
}

def load_dataset(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or Excel file.")

def clean_signal(series):
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.interpolate(limit_direction="both").dropna().to_numpy(dtype=float)
    if len(x) < 128:
        raise ValueError("Selected EEG channel needs at least 128 valid samples.")
    if not np.isfinite(x).all():
        raise ValueError("Signal contains invalid values that could not be cleaned.")
    return x

def preprocess(x, fs, lowcut, highcut, notch):
    y = x - np.mean(x)
    nyq = fs / 2
    upper = min(highcut, nyq * 0.95)
    lower = max(0.1, lowcut)
    if lower >= upper:
        raise ValueError("Sampling frequency is too low for the selected filter range.")
    sos = butter(4, [lower, upper], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, y)
    if notch and fs > 100:
        b, a = iirnotch(50, 30, fs)
        y = filtfilt(b, a, y)
    return y

def band_powers(x, fs):
    nperseg = min(len(x), max(128, int(fs * 4)))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)
    result = {}
    for label, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        result[label] = float(trapezoid(psd[mask], freqs[mask])) if mask.sum() > 1 else 0.0
    total = sum(result.values())
    relative = {k: (v / total * 100 if total > 0 else 0.0) for k, v in result.items()}
    return freqs, psd, result, relative

def make_windows(x, fs, seconds=2):
    size = max(32, int(fs * seconds))
    chunks = [x[i:i+size] for i in range(0, len(x)-size+1, size)]
    return chunks

def anomaly_analysis(x, fs, contamination):
    chunks = make_windows(x, fs)
    if len(chunks) < 5:
        return None, None, "At least 5 complete 2-second windows are needed for anomaly screening."
    feature_rows = []
    for chunk in chunks:
        _, _, powers, _ = band_powers(chunk, fs)
        feature_rows.append([
            np.std(chunk),
            np.sqrt(np.mean(chunk**2)),
            np.ptp(chunk),
            np.mean(np.abs(np.diff(chunk))),
            *powers.values()
        ])
    features = np.asarray(feature_rows)
    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42
    )
    labels = model.fit_predict(features)
    return labels, features, None

def build_pdf(report):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("NeuroScan AI – EEG Signal Analysis Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"Generated: {report['generated']}", styles["Normal"]),
        Paragraph(f"Dataset: {report['filename']}", styles["Normal"]),
        Paragraph(f"Channel: {report['channel']} | Sampling frequency: {report['fs']} Hz", styles["Normal"]),
        Spacer(1, 14),
        Paragraph("Signal summary", styles["Heading2"]),
    ]
    summary = [
        ["Measure", "Value"],
        ["Samples analysed", f"{report['samples']:,}"],
        ["Duration", f"{report['duration']:.2f} seconds"],
        ["Mean", f"{report['mean']:.4f}"],
        ["Standard deviation", f"{report['std']:.4f}"],
        ["RMS amplitude", f"{report['rms']:.4f}"],
        ["Peak-to-peak amplitude", f"{report['ptp']:.4f}"],
    ]
    t = Table(summary, colWidths=[220, 220])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("PADDING", (0,0), (-1,-1), 7),
    ]))
    story += [t, Spacer(1, 14), Paragraph("EEG frequency-band power", styles["Heading2"])]
    band_table = [["Frequency band", "Absolute power", "Relative power"]]
    for key in BANDS:
        band_table.append([key, f"{report['absolute'][key]:.6g}", f"{report['relative'][key]:.2f}%"])
    bt = Table(band_table, colWidths=[190, 130, 120])
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("PADDING", (0,0), (-1,-1), 7),
    ]))
    story += [bt, Spacer(1, 14), Paragraph("AI anomaly screening", styles["Heading2"])]
    story.append(Paragraph(report["anomaly_text"], styles["Normal"]))
    story += [Spacer(1, 14), Paragraph("Interpretation and limitations", styles["Heading2"])]
    story.append(Paragraph(
        "This report provides exploratory signal-processing and unsupervised anomaly-screening results only. "
        "Frequency-band power depends on montage, reference, acquisition hardware, filters, sampling rate and artifact quality. "
        "An anomaly is not a seizure, epilepsy diagnosis or evidence of disease. Results must not be used for clinical decisions.",
        styles["Normal"]
    ))
    doc.build(story)
    buffer.seek(0)
    return buffer

st.title("🧠 NeuroScan AI")
st.caption("EEG Signal Analysis • Frequency Bands • AI Anomaly Screening • PDF Report")
st.info("Research and education prototype. Upload de-identified EEG data only. This tool does not diagnose neurological conditions.")

with st.sidebar:
    st.header("Analysis settings")
    fs = st.number_input("Sampling frequency (Hz)", min_value=50, max_value=5000, value=256, step=1)
    lowcut = st.number_input("Bandpass low cutoff (Hz)", min_value=0.1, max_value=20.0, value=0.5, step=0.1)
    highcut = st.number_input("Bandpass high cutoff (Hz)", min_value=10.0, max_value=200.0, value=45.0, step=1.0)
    notch = st.checkbox("Apply 50 Hz notch filter", value=False)
    contamination = st.slider("Expected anomaly fraction (AI setting)", 0.01, 0.20, 0.05, 0.01)

uploaded = st.file_uploader("Upload EEG dataset", type=["csv", "xlsx", "xls"], help="Use one column per EEG channel. Optional time column is ignored during signal analysis.")
if uploaded is None:
    st.markdown("### Getting started")
    st.write("1. Upload a CSV or Excel EEG recording.")
    st.write("2. Select a numeric EEG channel.")
    st.write("3. Set the correct sampling frequency from the recording metadata.")
    st.write("4. Run analysis to view waveforms, frequency bands, spectrogram and anomaly windows.")
    st.download_button(
        "Download sample EEG dataset",
        data=open("sample_datasets/sample_eeg.csv", "rb").read(),
        file_name="sample_eeg.csv",
        mime="text/csv"
    )
    st.stop()

try:
    df = load_dataset(uploaded)
    if df.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()
    numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 128]
    # Time/index columns are not EEG signal channels; exclude them from channel selection.
    time_names = {"time", "time_s", "timestamp", "timestamps", "sample", "sample_index", "index"}
    eeg_cols = [c for c in numeric_cols if str(c).strip().lower() not in time_names]
    if eeg_cols:
        numeric_cols = eeg_cols
    if not numeric_cols:
        st.error("No numeric EEG channel with at least 128 valid samples was found.")
        st.stop()

    st.subheader("Dataset preview")
    st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
    st.dataframe(df.head(10), use_container_width=True)

    channel = st.selectbox("Choose EEG channel", numeric_cols)
    run = st.button("🔍 Analyze EEG Signal", type="primary", use_container_width=True)

    if run:
        raw = clean_signal(df[channel])
        processed = preprocess(raw, float(fs), float(lowcut), float(highcut), notch)
        freqs, psd, absolute, relative = band_powers(processed, float(fs))
        duration = len(processed) / float(fs)
        labels, features, anomaly_error = anomaly_analysis(processed, float(fs), contamination)

        st.session_state["analysis"] = {
            "filename": uploaded.name,
            "channel": str(channel),
            "fs": float(fs),
            "raw": raw,
            "processed": processed,
            "freqs": freqs,
            "psd": psd,
            "absolute": absolute,
            "relative": relative,
            "duration": duration,
            "labels": labels,
            "anomaly_error": anomaly_error,
            "mean": float(np.mean(processed)),
            "std": float(np.std(processed)),
            "rms": float(np.sqrt(np.mean(processed**2))),
            "ptp": float(np.ptp(processed)),
        }

except Exception as e:
    st.error(f"Could not read or analyse this dataset: {e}")

result = st.session_state.get("analysis")
if result:
    st.divider()
    st.subheader("Signal overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples", f"{len(result['processed']):,}")
    m2.metric("Duration", f"{result['duration']:.2f} s")
    m3.metric("Sampling rate", f"{result['fs']:.0f} Hz")
    m4.metric("Signal RMS", f"{result['rms']:.4g}")

    tabs = st.tabs(["EEG waveform", "Frequency analysis", "Spectrogram", "AI anomaly screening", "PDF report"])

    with tabs[0]:
        seconds_to_show = min(10, result["duration"])
        n = max(1, int(seconds_to_show * result["fs"]))
        t = np.arange(n) / result["fs"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=result["raw"][:n], name="Raw", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=t, y=result["processed"][:n], name="Filtered", line=dict(width=1)))
        fig.update_layout(title="EEG waveform (first 10 seconds or less)", xaxis_title="Time (s)", yaxis_title="Amplitude (input units)", template="plotly_dark", height=430)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(x=result["freqs"], y=result["psd"], mode="lines", fill="tozeroy"))
            fig.update_layout(title="Power spectral density", xaxis_title="Frequency (Hz)", yaxis_title="Power spectral density", template="plotly_dark", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(go.Bar(x=list(result["relative"].keys()), y=list(result["relative"].values())))
            fig.update_layout(title="Relative frequency-band power", yaxis_title="Relative power (%)", template="plotly_dark", height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame({
            "Band": list(result["absolute"].keys()),
            "Absolute power": list(result["absolute"].values()),
            "Relative power (%)": list(result["relative"].values())
        }), use_container_width=True)

    with tabs[2]:
        f, t, sxx = spectrogram(result["processed"], fs=result["fs"], nperseg=min(256, len(result["processed"])), noverlap=min(192, max(0, min(256, len(result["processed"]))-1)))
        fig = go.Figure(go.Heatmap(x=t, y=f, z=10*np.log10(sxx + 1e-12), colorscale="Viridis", colorbar_title="dB"))
        fig.update_layout(title="EEG spectrogram", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        if result["labels"] is None:
            st.warning(result["anomaly_error"])
        else:
            anomaly_count = int(np.sum(result["labels"] == -1))
            total_windows = len(result["labels"])
            st.metric("Flagged unusual windows", f"{anomaly_count}/{total_windows}")
            st.caption("Isolation Forest is an unsupervised screening method. Flags can reflect artifacts, movement, electrode noise or unusual physiology.")
            rows = []
            for i, label in enumerate(result["labels"]):
                rows.append({"Window": i+1, "Start time (s)": round(i*2, 2), "End time (s)": round((i+1)*2, 2), "Screening result": "Unusual pattern" if label == -1 else "Not flagged"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            fig = go.Figure(go.Bar(x=[r["Start time (s)"] for r in rows], y=[1 if r["Screening result"] == "Unusual pattern" else 0 for r in rows]))
            fig.update_layout(title="AI anomaly flags by 2-second window", xaxis_title="Window start (s)", yaxis_title="Flag (1 = unusual)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        anomaly_text = (
            f"Isolation Forest flagged {int(np.sum(result['labels'] == -1))} of {len(result['labels'])} windows."
            if result["labels"] is not None
            else result["anomaly_error"]
        )
        report = {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "filename": result["filename"],
            "channel": result["channel"],
            "fs": result["fs"],
            "samples": len(result["processed"]),
            "duration": result["duration"],
            "mean": result["mean"],
            "std": result["std"],
            "rms": result["rms"],
            "ptp": result["ptp"],
            "absolute": result["absolute"],
            "relative": result["relative"],
            "anomaly_text": anomaly_text,
        }
        pdf = build_pdf(report)
        st.success("Your analysis report is ready.")
        st.download_button(
            "📄 Download EEG Analysis PDF",
            data=pdf,
            file_name="NeuroScan_EEG_Analysis_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    st.caption("Exploratory analysis only. No clinical diagnosis or treatment recommendation is provided.")
