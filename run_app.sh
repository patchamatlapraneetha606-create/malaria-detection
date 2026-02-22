#!/bin/bash
# Run Malaria Detection app locally with training enabled
cd "$(dirname "$0")"
export ENABLE_LOCAL_TRAINING=1
.venv/bin/streamlit run app.py
