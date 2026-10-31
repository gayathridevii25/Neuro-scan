# NeuroScan AI – EEG Signal Analyzer

A Streamlit research/education prototype for EEG CSV/Excel uploads, signal preprocessing, frequency-band power analysis, waveform/spectrogram visualization, unsupervised anomaly screening, and PDF reporting.

## Features
- Upload CSV, XLSX, XLS EEG recordings
- Select a numeric EEG channel
- Configure sampling frequency and filters
- View raw and filtered waveform
- Welch power spectral density and relative Delta/Theta/Alpha/Beta/Gamma band power
- Spectrogram
- Isolation Forest anomaly screening on 2-second windows
- Download a PDF analysis report
- Sample synthetic EEG CSV included

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Dataset format
Use one column per EEG channel, with one sample per row. Example:

| time_s | Fp1 | Fp2 | Cz |
|---:|---:|---:|---:|
| 0.0000 | ... | ... | ... |
| 0.0039 | ... | ... | ... |

Set the sampling frequency to match the original recording metadata. The optional time column is not automatically used to infer sampling frequency. Do not mix channels with different sampling rates in one file.

## Suitable public datasets
1. **PhysioNet EEG Motor Movement/Imagery Database** – https://physionet.org/content/eegmmidb/1.0.0/  
   Public EEG recordings in EDF format. Convert selected channels to CSV before uploading; verify the dataset's license/terms and citation requirements.
2. **PhysioNet CHB-MIT Scalp EEG Database** – https://physionet.org/content/chbmit/1.0.0/  
   Scalp EEG recordings in EDF format. For research use; recordings contain sensitive health-related information. Use only de-identified data and respect PhysioNet access and citation conditions.
3. **OpenNeuro** – https://openneuro.org/  
   Search for EEG datasets; formats and access terms vary. Convert EEG channels to CSV for this app.

The app currently accepts CSV and Excel, not EDF directly. For EDF conversion, use MNE-Python locally:
```python
import mne
raw = mne.io.read_raw_edf("recording.edf", preload=True, verbose=False)
data = raw.get_data().T
df = pd.DataFrame(data, columns=raw.ch_names)
df.insert(0, "time_s", raw.times)
df.to_csv("recording.csv", index=False)
```
Install conversion dependencies separately with `pip install mne`.

## GitHub + Streamlit Cloud deployment (Android phone)
1. Download and extract the project ZIP.
2. Open https://github.com in Chrome and sign in.
3. Create a repository named `NeuroScan-AI`.
4. Upload `app.py`, `requirements.txt`, `README.md`, and optionally `sample_datasets/sample_eeg.csv` (upload files using the repository's Add file → Upload files interface).
5. Open https://share.streamlit.io/ and sign in with GitHub.
6. Choose **Create app**, select repository `NeuroScan-AI`, branch `main`, and main file `app.py`.
7. Deploy. Wait for dependency installation and app startup.
8. Open the deployed app URL on your phone. Use the sample CSV first to test the workflow.

## Important deployment notes
- Streamlit Community Cloud has resource, execution-time and storage limits. Large EEG files may exceed them.
- Uploaded files are processed in app memory; this starter app does not permanently store uploads.
- Do not upload identifiable patient records or private clinical data to a public GitHub repository or public demo.
- If the app fails to start, inspect the Streamlit Cloud app logs and confirm all files are in the repository root as shown.

## Method and limitations
- Filtering: 4th-order Butterworth bandpass with zero-phase filtering; optional 50 Hz notch.
- Spectral analysis: Welch PSD and band integration.
- AI: Isolation Forest on window-level signal features. It is unsupervised and does not classify seizures or diagnose disease.
- The Delta/Theta/Alpha/Beta/Gamma bands are approximate conventional ranges; definitions vary by protocol.
- The app assumes a selected channel's values are numeric and evenly sampled. The correct sampling frequency must be entered from acquisition metadata.
- Filter choices, channel montage/reference, artifacts, units and recording hardware affect results.
- This is an educational/research prototype, not a medical device. Do not use it for diagnosis, triage or treatment decisions.

## Suggested citation
When using public datasets, cite the original dataset paper and repository page as required by its terms.
