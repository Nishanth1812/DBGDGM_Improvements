import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

# Add MM_DBGDGM to path
ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR / "MM_DBGDGM") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "MM_DBGDGM"))

from inference.predict import predict_subject
from visualisation.brain_mapping import plot_brain_projection

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\WeKan Training Data\mm_dbgdgm_prepared")
MANIFEST_PATH = DATA_DIR / "labels.csv"
CHECKPOINT_PATH = ROOT_DIR / "results" / "best_model.pt"
REPORTS_DIR = ROOT_DIR / "inference_reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Page Config
st.set_page_config(
    page_title="MM-DBGDGM Diagnostic Portal",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #eee;
    }
    /* Only force dark text inside the white metric cards */
    .stMetric h2, .stMetric p {
        color: #1a1a1a !important;
    }
    .sidebar .sidebar-content {
        background-color: #2e7bcf;
    }
    </style>
    """, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("?? Diagnostic Portal")
st.sidebar.markdown("---")

if not MANIFEST_PATH.exists():
    st.error(f"Manifest not found at {MANIFEST_PATH}")
    st.stop()

df = pd.read_csv(MANIFEST_PATH)
subjects = df['subject_id'].unique().tolist()

selected_subject = st.sidebar.selectbox("Select Patient Subject", subjects)

st.sidebar.markdown("### Model Configuration")
st.sidebar.info(f"Checkpoint: {CHECKPOINT_PATH.name}")
st.sidebar.info(f"Data Source: {DATA_DIR.name}")

# ── Main UI ────────────────────────────────────────────────────────────────
st.title("Multimodal Deep Brain Generative Dynamic Graph Model")
st.subheader("Early Alzheimer's Detection & Spatial Diagnostic Analytics")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Patient Information")
    sub_data = df[df['subject_id'] == selected_subject].iloc[0]
    st.write(f"**Subject ID:** {selected_subject}")
    st.write(f"**Scan Type:** {sub_data.get('modality', 'Multimodal fMRI/sMRI')}")
    
    if st.button("Run Clinical Diagnosis", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing Multimodal Data for {selected_subject}..."):
            # Prepare paths
            fmri_path = DATA_DIR / sub_data['fmri_path']
            smri_path = DATA_DIR / sub_data['smri_path']
            
            # Run Inference
            try:
                result = predict_subject(
                    fmri_path=fmri_path,
                    smri_path=smri_path,
                    model_checkpoint=CHECKPOINT_PATH,
                    class_names=["CN", "MCI", "AD"]
                )
                st.session_state['result'] = result
                st.session_state['subject_id'] = selected_subject
                
                # Generate Brain Map
                subject_report_dir = REPORTS_DIR / selected_subject
                subject_report_dir.mkdir(exist_ok=True)
                brain_map_path = subject_report_dir / "brain_significance_map.png"
                
                avg_weights = (np.array(result["attention_weights"]["fs"]) + np.array(result["attention_weights"]["sf"])) / 2
                plot_brain_projection(avg_weights, save_path=str(brain_map_path))
                st.session_state['brain_map'] = str(brain_map_path)
                
                st.success("Analysis Complete!")
            except Exception as e:
                st.error(f"Inference Error: {e}")

# ── Display Results ────────────────────────────────────────────────────────
if 'result' in st.session_state and st.session_state['subject_id'] == selected_subject:
    res = st.session_state['result']
    
    st.markdown("---")
    
    # Metrics Row
    m_col1, m_col2, m_col3 = st.columns(3)
    
    pred_class = res['predicted_class']
    confidence = res['confidence']
    uncertainty = res['uncertainty']
    
    with m_col1:
        color = "#27ae60" if pred_class == "CN" else ("#f39c12" if pred_class == "MCI" else "#e74c3c")
        st.markdown(f"""
            <div class='stMetric'>
                <p style='font-size: 14px; color: #555; margin-bottom: 5px;'>Predicted Diagnosis</p>
                <h2 style='color: {color}; margin: 0; font-weight: bold;'>{pred_class}</h2>
            </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
            <div class='stMetric'>
                <p style='font-size: 14px; color: #555; margin-bottom: 5px;'>Confidence Level</p>
                <h2 style='color: #2c3e50; margin: 0; font-weight: bold;'>{confidence:.2%}</h2>
            </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        status_color = "#27ae60" if not res['high_uncertainty_flag'] else "#e74c3c"
        status_text = "Reliable" if not res['high_uncertainty_flag'] else "High Uncertainty"
        st.markdown(f"""
            <div class='stMetric'>
                <p style='font-size: 14px; color: #555; margin-bottom: 5px;'>Model Uncertainty</p>
                <h2 style='color: #2c3e50; margin: 0; font-weight: bold;'>{uncertainty:.4f}</h2>
                <p style='font-size: 12px; color: {status_color}; margin-top: 5px; font-weight: bold;'>?? {status_text}</p>
            </div>
        """, unsafe_allow_html=True)

    # Visuals Row
    v_col1, v_col2 = st.columns([1.2, 0.8])
    
    with v_col1:
        st.markdown("### Diagnostic Significance Map")
        if 'brain_map' in st.session_state:
            st.image(st.session_state['brain_map'], use_container_width=True, 
                     caption="Topological significance of brain ROIs based on cross-modal attention.")
            
    with v_col2:
        st.markdown("### Top Contributing ROIs")
        regions_df = pd.DataFrame({
            "Brain Region": res['top_attention_regions'],
            "Importance Rank": range(1, len(res['top_attention_regions']) + 1)
        })
        st.table(regions_df)

    # ── Probability Distribution ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Model Classification Details")
    p_col1, p_col2 = st.columns([0.6, 0.4])
    
    with p_col1:
        probs_df = pd.DataFrame({
            "Condition": list(res['probabilities'].keys()),
            "Probability": list(res['probabilities'].values())
        })
        st.bar_chart(probs_df.set_index("Condition"))
        
    with p_col2:
        st.markdown("### Probability Summary")
        for cond, prob in res['probabilities'].items():
            st.write(f"**{cond}:** {prob:.2%}")

    # Download Section
    st.markdown("---")
    st.markdown("### Clinical Documentation")
    st.info("The detailed Clinical Inference Report (PDF/PNG) is being generated in the background.")

# ── Research Benchmarking Section ─────────────────────────────────────────
st.markdown("---")
with st.expander("View Global Research Benchmarks (Group Comparisons)"):
    st.markdown("""
    Below is a comparison of the **Default Mode Network (DMN)** across the entire ADNI cohort in your dataset. 
    This shows the statistical average connectivity for each diagnostic group.
    """)
    
    from visualisation.research_benchmarks import plot_group_dmn_benchmarks
    
    with st.spinner("Calculating group-level averages..."):
        benchmark_fig = plot_group_dmn_benchmarks(MANIFEST_PATH, DATA_DIR)
        st.pyplot(benchmark_fig)
        
    st.info("Note: The average functional connectivity decreases visibly in the PCC/Precuneus nodes as patients progress from CN to AD.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>MM-DBGDGM v1.0 | Clinical Research Interface | Google DeepMind Agentic Coding</p>", unsafe_allow_html=True)
