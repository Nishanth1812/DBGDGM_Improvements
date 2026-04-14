"""
Clinical Report Generator
Takes the deep inference output dictionary from the MM-DBGDGM and generates
a structured, explainable clinical narrative for subject staging and prediction.
"""

from typing import Dict, List

STAGE_MAP = {
    0: "Cognitively Normal (CN)",
    1: "Early Mild Cognitive Impairment (eMCI)",
    2: "Late Mild Cognitive Impairment (lMCI)",
    3: "Alzheimer's Disease (AD)"
}

def generate_subject_report(subject_id: str, clinical_data: Dict) -> str:
    """
    Formats the dict returned from `MM_DBGDGM.predict()` into a 
    physician-readable markdown report.
    
    Args:
        subject_id: Identifier for the patient
        clinical_data: dict from MM_DBGDGM.predict()
    """
    current_stage = STAGE_MAP.get(clinical_data['current_stage_prediction'], "Unknown")
    probs = clinical_data['stage_probabilities']
    
    # 1. Uncertainty Warning
    uncertain_flag = ""
    if clinical_data.get('uncertainty_flag', False):
        uncertain_flag = (
            "🚨 **CLINICAL REVIEW REQUIRED** 🚨\n"
            f"The model has flagged high uncertainty for this prediction (Posterior Variance: {clinical_data['mean_posterior_variance']:.3f}). "
            "Please review carefully.\n\n"
        )
    
    # 2. Stage classification
    stage_report = (
        f"### Disease Staging\n"
        f"**Predicted Stage:** {current_stage}\n"
        f"**Confidence Profile:**\n"
        f"- CN:   {probs[0]*100:.1f}%\n"
        f"- eMCI: {probs[1]*100:.1f}%\n"
        f"- lMCI: {probs[2]*100:.1f}%\n"
        f"- AD:   {probs[3]*100:.1f}%\n"
    )
    
    # 3. Deterioration/Regression predictions
    nss = clinical_data.get('nss_score', 0.0)
    hippo = clinical_data.get('hippocampal_volume', 0.0)
    thinning = clinical_data.get('cortical_thinning_rate', 0.0)
    
    regression_report = (
        f"### Biomarker Forecasts\n"
        f"- **Neurodegeneration Severity Score (NSS, 0-100):** {nss:.2f}\n"
        f"- **Predicted Future Hippocampal Volume:** {hippo:.1f} mm³\n"
        f"- **Estimated Cortical Thinning Rate:** {thinning:.4f} mm/year\n"
    )
    
    # 4. Regional Atrophy
    # Assuming indices correspond to some predefined atlas (e.g. AAL-90 or Schaefer-200)
    # We will just list the Top-5 regional indices.
    loc_scores = clinical_data.get('atrophy_localization_scores', [])
    if len(loc_scores) > 0:
        top_k = 5
        top_indices = sorted(range(len(loc_scores)), key=lambda i: loc_scores[i], reverse=True)[:top_k]
        regions_str = "\n".join([f"- ROI {idx} (Score: {loc_scores[idx]:.3f})" for idx in top_indices])
        atrophy_report = f"### Top {top_k} Regions Vulnerable to Atrophy\n{regions_str}\n"
    else:
        atrophy_report = ""
        
    # 5. Survival/Conversion Predictions
    survival = clinical_data.get('expected_time_to_events', [])
    if len(survival) == 3:
        conversion_report = (
            f"### Disease Progression Estimates\n"
            f"- **Expected Time to eMCI:** {survival[0]:.1f} years\n"
            f"- **Expected Time to lMCI:** {survival[1]:.1f} years\n"
            f"- **Expected Time to AD:**   {survival[2]:.1f} years\n"
            f"*(Note: These estimates assume an uninterrupted natural history of progression.)*\n"
        )
    else:
        conversion_report = ""
        
    # Compile full report
    full_report = (
        f"## MM-DBGDGM Clinical Subject Report: {subject_id}\n\n"
        f"{uncertain_flag}"
        f"{stage_report}\n"
        f"{regression_report}\n"
        f"{atrophy_report}\n"
        f"{conversion_report}\n"
        f"---\nDisclaimer: This is an AI-generated investigative report for preclinical research."
    )
    
    return full_report
