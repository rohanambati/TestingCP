# Warp Project Guide

## Project Overview

This repository contains a Streamlit web application for detecting fake/automated social media accounts using multiple ML/DL models (BiLSTM-Att, GCN, TGN, MMT, CTPP-GNN, and XGBoost). The main entrypoint is `app.py`.

## Main Entry Points

- **Streamlit app**: `app.py`
- **Dependencies**: `requirements.txt`
- **Model artifacts expected** (by default):
  - `best_bilstm_attention_model.pth`
  - `best_ctpp_gnn_model.pth`
  - `best_gcn_model.pth`
  - `best_mmt_model.pth`
  - `best_tgn_model.pth`
  - `cnn_model.pth`
  - `xgb_model_tuned.json`

> Note: In `app.py`, these are loaded from a hardcoded `output_path` (`/content/drive/MyDrive/Capstone Project New/models/`). If you want to run locally with files in the repo root, you will need to adjust `output_path` accordingly.

## How to Run the App

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Adjust model path (optional but recommended for local)**

   In `app.py`, update the `output_path` to where your model files actually live, for example:

   ```python
   output_path = "./"  # or an absolute/relative path containing the .pth and .json files
   ```

3. **Ensure model files are present**

   Place all required model files into the directory pointed to by `output_path`.

4. **Run Streamlit**

   ```bash
   streamlit run app.py
   ```

## Features & Flows

- **Manual Text & Metadata Input**
  - Enter a bio text.
  - Provide numerical metadata (followers, following, posts, etc.).
  - Choose a model and click **"Classify Profile"**.

- **Reddit Profile Analysis**
  - Requires environment variables: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`.
  - Enter a Reddit username and fetch data; then classify using any supported model.

- **Ngrok Integration (optional)**
  - If `NGROK_AUTH_TOKEN` is set in the environment, the app will attempt to expose your local Streamlit server via ngrok and display the public URL.

## Testing & Validation

There is no dedicated test suite in this repository. To validate changes:

- Run `streamlit run app.py` and exercise both input modes.
- Verify that each model type can be selected and used without raising errors (assuming its weights are present under `output_path`).

## Common Gotchas

- **Missing `output_path` directory**: If the configured `output_path` does not exist, the app will raise an error and stop early. Update `output_path` or create the directory and place model files there.
- **Missing model files**: `load_model()` and `load_cnn_model()` expect specific filenames; mismatched or missing files will cause loading errors.
- **Reddit credentials not set**: The Reddit analysis path requires PRAW credentials in environment variables.
- **Colab-specific paths**: The default `output_path` is designed for Google Colab + Google Drive; adjust for local development.

## How Warp Agent Can Help

- Refactor `app.py` into smaller modules (e.g., `models/`, `data/`, `ui/`).
- Add or update model architectures and loading logic.
- Modify `output_path` handling to be configurable (e.g., via environment variable or Streamlit sidebar setting).
- Add tests or simple health checks for model loading and inference.
