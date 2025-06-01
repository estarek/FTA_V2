import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set page configuration
st.set_page_config(
    page_title="E-Invoice Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
MODEL_DIR = "model_artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_detection_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
FEATURE_METADATA_PATH = os.path.join(MODEL_DIR, "feature_metadata.json")
EVALUATION_METRICS_PATH = os.path.join(MODEL_DIR, "evaluation_metrics.json")
RISK_SCORED_INVOICES_PATH = os.path.join(MODEL_DIR, "risk_scored_invoices.csv")
TOP_RISK_INVOICES_PATH = os.path.join(MODEL_DIR, "top_risk_invoices.csv")
BOTTOM_RISK_INVOICES_PATH = os.path.join(MODEL_DIR, "bottom_risk_invoices.csv")
ANOMALY_TYPE_DISTRIBUTION_PATH = os.path.join(MODEL_DIR, "anomaly_type_distribution.csv")
EMIRATE_DISTRIBUTION_PATH = os.path.join(MODEL_DIR, "emirate_distribution.csv")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E3A8A;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
    .high-risk {
        color: #DC2626;
        font-weight: bold;
    }
    .medium-risk {
        color: #F59E0B;
        font-weight: bold;
    }
    .low-risk {
        color: #10B981;
        font-weight: bold;
    }
    .info-box {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
    .error-box {
        background-color: #FEE2E2;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #DC2626;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    """Load all necessary data and model artifacts with robust error handling"""
    data = {}
    missing_files = []
    
    # Define all files to load
    files_to_load = {
        "model": MODEL_PATH,
        "scaler": SCALER_PATH,
        "label_encoders": LABEL_ENCODERS_PATH,
        "feature_metadata": FEATURE_METADATA_PATH,
        "evaluation_metrics": EVALUATION_METRICS_PATH,
        "risk_scored_invoices": RISK_SCORED_INVOICES_PATH,
        "top_risk_invoices": TOP_RISK_INVOICES_PATH,
        "bottom_risk_invoices": BOTTOM_RISK_INVOICES_PATH,
        "anomaly_type_distribution": ANOMALY_TYPE_DISTRIBUTION_PATH,
        "emirate_distribution": EMIRATE_DISTRIBUTION_PATH
    }
    
    # Check if model_artifacts directory exists
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model artifacts directory '{MODEL_DIR}' not found. Please ensure it exists in the same directory as this script.")
        return None
    
    # Check which files exist
    for key, file_path in files_to_load.items():
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        st.error(f"The following required files are missing: {', '.join(missing_files)}")
        return None
    
    # Load files with appropriate error handling
    try:
        # Load model
        try:
            data["model"] = joblib.load(MODEL_PATH)
            # Extract feature importances immediately to avoid caching issues
            if hasattr(data["model"], 'feature_importances_'):
                data["feature_importances"] = data["model"].feature_importances_
            else:
                data["feature_importances"] = None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            data["model"] = None
            data["feature_importances"] = None
        
        # Load scaler
        try:
            data["scaler"] = joblib.load(SCALER_PATH)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            data["scaler"] = None
        
        # Load label encoders
        try:
            with open(LABEL_ENCODERS_PATH, 'rb') as f:
                data["label_encoders"] = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading label encoders: {e}")
            data["label_encoders"] = None
        
        # Load feature metadata
        try:
            with open(FEATURE_METADATA_PATH, 'r') as f:
                data["feature_metadata"] = json.load(f)
        except Exception as e:
            st.error(f"Error loading feature metadata: {e}")
            data["feature_metadata"] = None
        
        # Load evaluation metrics
        try:
            with open(EVALUATION_METRICS_PATH, 'r') as f:
                data["evaluation_metrics"] = json.load(f)
        except Exception as e:
            st.error(f"Error loading evaluation metrics: {e}")
            data["evaluation_metrics"] = None
        
        # Load CSV files with error handling
        csv_files = {
            "risk_scored_invoices": RISK_SCORED_INVOICES_PATH,
            "top_risk_invoices": TOP_RISK_INVOICES_PATH,
            "bottom_risk_invoices": BOTTOM_RISK_INVOICES_PATH,
            "anomaly_type_distribution": ANOMALY_TYPE_DISTRIBUTION_PATH,
            "emirate_distribution": EMIRATE_DISTRIBUTION_PATH
        }
        
        for key, file_path in csv_files.items():
            try:
                data[key] = pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Error loading {key}: {e}")
                # Create empty DataFrame with expected columns as fallback
                if key == "risk_scored_invoices" or key == "top_risk_invoices" or key == "bottom_risk_invoices":
                    data[key] = pd.DataFrame(columns=["invoice_number", "true_anomaly", "anomaly_risk_score", 
                                                     "predicted_anomaly", "original_anomaly_type", "original_explanation"])
                elif key == "anomaly_type_distribution":
                    data[key] = pd.DataFrame(columns=["anomaly_type", "count"])
                elif key == "emirate_distribution":
                    data[key] = pd.DataFrame(columns=["emirate", "count"])
        
        return data
    
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        return None

@st.cache_data
def plot_risk_score_distribution(risk_scored_invoices_df):
    """Plot the distribution of risk scores with error handling"""
    try:
        if risk_scored_invoices_df is None or len(risk_scored_invoices_df) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for risk score distribution",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Distribution of Risk Scores (No Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Check if required columns exist
        required_cols = ["anomaly_risk_score", "true_anomaly"]
        if not all(col in risk_scored_invoices_df.columns for col in required_cols):
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required columns for risk score distribution",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Distribution of Risk Scores (Missing Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Ensure true_anomaly is numeric
        risk_scored_invoices = risk_scored_invoices_df.copy()
        risk_scored_invoices["true_anomaly"] = pd.to_numeric(risk_scored_invoices["true_anomaly"], errors="coerce").fillna(0).astype(int)
        
        fig = px.histogram(
            risk_scored_invoices, 
            x="anomaly_risk_score",
            color="true_anomaly",
            nbins=50,
            labels={"anomaly_risk_score": "Risk Score", "true_anomaly": "Is Anomaly"},
            color_discrete_map={0: "#10B981", 1: "#DC2626"},
            title="Distribution of Risk Scores"
        )
        
        fig.update_layout(
            xaxis_title="Risk Score",
            yaxis_title="Count",
            legend_title="Is Anomaly",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting risk score distribution: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Distribution of Risk Scores (Error)",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

@st.cache_data
def plot_confusion_matrix(confusion_matrix_data):
    """Plot the confusion matrix with robust error handling"""
    try:
        if confusion_matrix_data is None:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No confusion matrix data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Confusion Matrix (No Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Convert to numpy array if it's a list
        if isinstance(confusion_matrix_data, list):
            cm = np.array(confusion_matrix_data)
        else:
            cm = confusion_matrix_data
        
        # Calculate percentages - handle both list and numpy array formats
        if isinstance(cm, np.ndarray):
            total = cm.sum()
            cm_percent = [[val / total * 100 for val in row] for row in cm]
        else:
            total = sum(sum(row) for row in cm)
            cm_percent = [[val / total * 100 for val in row] for row in cm]
        
        # Create annotation text
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)",
                        font=dict(color="white" if (i == j) else "black"),
                        showarrow=False,
                    )
                )
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Normal", "Predicted Anomaly"],
            y=["Actual Normal", "Actual Anomaly"],
            colorscale=[[0, "#10B981"], [1, "#DC2626"]],
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            annotations=annotations,
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting confusion matrix: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Confusion Matrix (Error)",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

@st.cache_data
def plot_feature_importance(feature_importances, feature_names):
    """Plot feature importance with error handling - using hashable parameters only"""
    try:
        if feature_importances is None or feature_names is None:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No feature importance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Feature Importance (No Data)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Check if lengths match
        if len(feature_importances) != len(feature_names):
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Feature importance length ({len(feature_importances)}) doesn't match feature names length ({len(feature_names)})",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Feature Importance (Dimension Mismatch)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Create a dataframe for plotting
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importances
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values("Importance", ascending=False).head(15)
        
        # Create the plot
        fig = px.bar(
            feature_importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Feature Importance",
            color="Importance",
            color_continuous_scale=px.colors.sequential.Blues,
        )
        
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting feature importance: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Feature Importance (Error)",
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

@st.cache_data
def plot_anomaly_type_distribution(anomaly_type_distribution_df):
    """Plot the distribution of anomaly types with error handling"""
    try:
        if anomaly_type_distribution_df is None or len(anomaly_type_distribution_df) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No anomaly type distribution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Anomaly Type Distribution (No Data)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Check if required columns exist
        required_cols = ["anomaly_type", "count"]
        if not all(col in anomaly_type_distribution_df.columns for col in required_cols):
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required columns for anomaly type distribution",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Anomaly Type Distribution (Missing Data)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Sort by count descending
        df_sorted = anomaly_type_distribution_df.sort_values("count", ascending=False)
        
        # Create the plot
        fig = px.bar(
            df_sorted,
            x="anomaly_type",
            y="count",
            title="Distribution of Anomaly Types",
            color="count",
            color_continuous_scale=px.colors.sequential.Reds,
        )
        
        fig.update_layout(
            xaxis_title="Anomaly Type",
            yaxis_title="Count",
            xaxis=dict(tickangle=45),
            height=500,
            margin=dict(l=40, r=40, t=40, b=80),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting anomaly type distribution: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Anomaly Type Distribution (Error)",
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

@st.cache_data
def plot_emirate_distribution(emirate_distribution_df):
    """Plot the distribution of invoices by emirate with error handling"""
    try:
        if emirate_distribution_df is None or len(emirate_distribution_df) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No emirate distribution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Emirate Distribution (No Data)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Check if required columns exist
        required_cols = ["emirate", "count"]
        if not all(col in emirate_distribution_df.columns for col in required_cols):
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required columns for emirate distribution",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Emirate Distribution (Missing Data)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Sort by count descending
        df_sorted = emirate_distribution_df.sort_values("count", ascending=False)
        
        # Create the plot
        fig = px.pie(
            df_sorted,
            values="count",
            names="emirate",
            title="Distribution of Invoices by Emirate",
            color_discrete_sequence=px.colors.sequential.Blues,
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting emirate distribution: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Emirate Distribution (Error)",
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

@st.cache_data
def plot_emirate_map(emirate_distribution_df):
    """Plot the UAE map with emirate distribution with error handling"""
    try:
        if emirate_distribution_df is None or len(emirate_distribution_df) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No emirate distribution data available for map",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="UAE Emirate Distribution Map (No Data)",
                height=600,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # UAE emirate coordinates (approximate centroids)
        emirate_coords = {
            "Abu Dhabi": (24.4539, 54.3773),
            "Dubai": (25.2048, 55.2708),
            "Sharjah": (25.3463, 55.4209),
            "Ajman": (25.4111, 55.4354),
            "Umm Al Quwain": (25.5647, 55.5534),
            "Ras Al Khaimah": (25.7895, 55.9432),
            "Fujairah": (25.1288, 56.3265)
        }
        
        # Create a dataframe with coordinates
        map_data = []
        for _, row in emirate_distribution_df.iterrows():
            emirate = row["emirate"]
            if emirate in emirate_coords:
                lat, lon = emirate_coords[emirate]
                map_data.append({
                    "emirate": emirate,
                    "count": row["count"],
                    "lat": lat,
                    "lon": lon
                })
        
        if not map_data:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No valid emirate data for mapping",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="UAE Emirate Distribution Map (No Valid Data)",
                height=600,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        map_df = pd.DataFrame(map_data)
        
        # Create the map
        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            size="count",
            color="count",
            hover_name="emirate",
            hover_data={"count": True, "lat": False, "lon": False},
            color_continuous_scale=px.colors.sequential.Blues,
            size_max=50,
            zoom=6,
            center={"lat": 24.7, "lon": 54.5},
            title="UAE Emirate Distribution Map"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            height=600,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting emirate map: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="UAE Emirate Distribution Map (Error)",
            height=600,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

@st.cache_data
def plot_risk_by_emirate(risk_scored_invoices_df):
    """Plot the average risk score by emirate with error handling"""
    try:
        if risk_scored_invoices_df is None or len(risk_scored_invoices_df) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No risk score data available for emirate analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Average Risk Score by Emirate (No Data)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Check if required columns exist
        if "buyer_emirate" not in risk_scored_invoices_df.columns or "anomaly_risk_score" not in risk_scored_invoices_df.columns:
            # Try to use a fallback approach
            if "original_anomaly_type" in risk_scored_invoices_df.columns:
                # Create a synthetic risk by emirate dataframe
                risk_by_emirate = pd.DataFrame({
                    "emirate": ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Umm Al Quwain", "Ras Al Khaimah", "Fujairah"],
                    "avg_risk_score": np.random.uniform(0.1, 0.3, 7)
                })
                
                fig = px.bar(
                    risk_by_emirate,
                    x="emirate",
                    y="avg_risk_score",
                    title="Average Risk Score by Emirate (Simulated Data)",
                    color="avg_risk_score",
                    color_continuous_scale=px.colors.sequential.Reds,
                )
                
                fig.update_layout(
                    xaxis_title="Emirate",
                    yaxis_title="Average Risk Score",
                    height=500,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                return fig
            else:
                # Return empty figure with message
                fig = go.Figure()
                fig.add_annotation(
                    text="Missing required columns for risk by emirate analysis",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(
                    title="Average Risk Score by Emirate (Missing Data)",
                    height=500,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                return fig
        
        # Group by emirate and calculate average risk score
        risk_by_emirate = risk_scored_invoices_df.groupby("buyer_emirate")["anomaly_risk_score"].mean().reset_index()
        risk_by_emirate.columns = ["emirate", "avg_risk_score"]
        
        # Sort by average risk score descending
        risk_by_emirate = risk_by_emirate.sort_values("avg_risk_score", ascending=False)
        
        # Create the plot
        fig = px.bar(
            risk_by_emirate,
            x="emirate",
            y="avg_risk_score",
            title="Average Risk Score by Emirate",
            color="avg_risk_score",
            color_continuous_scale=px.colors.sequential.Reds,
        )
        
        fig.update_layout(
            xaxis_title="Emirate",
            yaxis_title="Average Risk Score",
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting risk by emirate: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Average Risk Score by Emirate (Error)",
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

def main():
    """Main function to run the Streamlit app"""
    # Load data
    data = load_data()
    
    # Header
    st.markdown('<h1 class="main-header">E-Invoice Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Info box
    st.markdown(
        '<div class="info-box">'
        '<p>This dashboard demonstrates advanced AI capabilities for detecting anomalies and potential fraud '
        'in e-invoice data. The model analyzes patterns across multiple dimensions to identify suspicious '
        'transactions and assign risk scores.</p>'
        '<p><strong>Note:</strong> This dashboard uses synthetic data generated for demonstration purposes.</p>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Check if data is loaded successfully
    if data is None:
        st.error("Failed to load data. Please check the console for error messages.")
        return
    
    # Create tabs
    tabs = st.tabs([
        "📊 Overview", 
        "🔍 Model Performance", 
        "🌍 Geographic Insights", 
        "📝 Invoice Explorer"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        st.markdown('<h2 class="sub-header">Fraud Detection Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                '<div class="metric-card">'
                '<p class="metric-label">Model Accuracy</p>'
                f'<p class="metric-value">{data["evaluation_metrics"]["accuracy"]:.2%}</p>'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            # Calculate precision from classification report
            precision = data["evaluation_metrics"]["classification_report"]["Anomaly"]["precision"]
            st.markdown(
                '<div class="metric-card">'
                '<p class="metric-label">Anomaly Precision</p>'
                f'<p class="metric-value">{precision:.2%}</p>'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col3:
            # Calculate recall from classification report
            recall = data["evaluation_metrics"]["classification_report"]["Anomaly"]["recall"]
            st.markdown(
                '<div class="metric-card">'
                '<p class="metric-label">Anomaly Recall</p>'
                f'<p class="metric-value">{recall:.2%}</p>'
                '</div>',
                unsafe_allow_html=True
            )
        
        # Risk score distribution
        st.markdown('<h3 class="sub-header">Risk Score Distribution</h3>', unsafe_allow_html=True)
        st.plotly_chart(plot_risk_score_distribution(data["risk_scored_invoices"]), use_container_width=True)
        
        # Anomaly type distribution
        st.markdown('<h3 class="sub-header">Anomaly Type Distribution</h3>', unsafe_allow_html=True)
        st.plotly_chart(plot_anomaly_type_distribution(data["anomaly_type_distribution"]), use_container_width=True)
    
    # Tab 2: Model Performance
    with tabs[1]:
        st.markdown('<h2 class="sub-header">Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown('<h3 class="sub-header">Confusion Matrix</h3>', unsafe_allow_html=True)
        st.plotly_chart(plot_confusion_matrix(data["evaluation_metrics"]["confusion_matrix"]), use_container_width=True)
        
        # Feature Importance
        st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
        
        # Check if feature_importances and feature_metadata are available
        if data["feature_importances"] is not None and data["feature_metadata"] is not None:
            # Get feature names from metadata
            feature_names = data["feature_metadata"]["valid_features"]
            
            # Check if lengths match
            if len(data["feature_importances"]) == len(feature_names):
                st.plotly_chart(plot_feature_importance(data["feature_importances"], feature_names), use_container_width=True)
            else:
                st.error(f"Feature importance length ({len(data['feature_importances'])}) doesn't match feature names length ({len(feature_names)})")
        else:
            st.error("Feature importance data not available")
        
        # Classification Report
        st.markdown('<h3 class="sub-header">Classification Report</h3>', unsafe_allow_html=True)
        
        if "classification_report" in data["evaluation_metrics"]:
            report = data["evaluation_metrics"]["classification_report"]
            
            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).drop("accuracy", errors="ignore")
            
            # Transpose for better display
            report_df = report_df.T
            
            # Format as percentages
            for col in ["precision", "recall", "f1-score"]:
                if col in report_df.columns:
                    report_df[col] = report_df[col].map("{:.2%}".format)
            
            st.dataframe(report_df, use_container_width=True)
        else:
            st.error("Classification report not available")
    
    # Tab 3: Geographic Insights
    with tabs[2]:
        st.markdown('<h2 class="sub-header">Geographic Insights</h2>', unsafe_allow_html=True)
        
        # Emirate Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">Invoice Distribution by Emirate</h3>', unsafe_allow_html=True)
            st.plotly_chart(plot_emirate_distribution(data["emirate_distribution"]), use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="sub-header">Average Risk Score by Emirate</h3>', unsafe_allow_html=True)
            st.plotly_chart(plot_risk_by_emirate(data["risk_scored_invoices"]), use_container_width=True)
        
        # UAE Map
        st.markdown('<h3 class="sub-header">UAE Emirate Distribution Map</h3>', unsafe_allow_html=True)
        st.plotly_chart(plot_emirate_map(data["emirate_distribution"]), use_container_width=True)
    
    # Tab 4: Invoice Explorer
    with tabs[3]:
        st.markdown('<h2 class="sub-header">Invoice Explorer</h2>', unsafe_allow_html=True)
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            risk_threshold = st.slider("Risk Score Threshold", 0.0, 1.0, 0.5, 0.05)
        
        with col2:
            if "original_anomaly_type" in data["risk_scored_invoices"].columns:
                anomaly_types = data["risk_scored_invoices"]["original_anomaly_type"].dropna().unique().tolist()
                selected_anomaly_type = st.selectbox("Filter by Anomaly Type", ["All"] + anomaly_types)
            else:
                selected_anomaly_type = "All"
        
        # Filter data
        filtered_data = data["risk_scored_invoices"].copy()
        
        # Apply risk threshold filter
        filtered_data = filtered_data[filtered_data["anomaly_risk_score"] >= risk_threshold]
        
        # Apply anomaly type filter if selected
        if selected_anomaly_type != "All" and "original_anomaly_type" in filtered_data.columns:
            filtered_data = filtered_data[filtered_data["original_anomaly_type"] == selected_anomaly_type]
        
        # Display filtered invoices
        st.markdown('<h3 class="sub-header">Filtered Invoices</h3>', unsafe_allow_html=True)
        
        if len(filtered_data) > 0:
            # Add risk category
            filtered_data["risk_category"] = pd.cut(
                filtered_data["anomaly_risk_score"],
                bins=[0, 0.7, 0.9, 1.0],
                labels=["Low", "Medium", "High"]
            )
            
            # Format for display
            display_df = filtered_data[["invoice_number", "anomaly_risk_score", "risk_category", "original_anomaly_type", "original_explanation"]].copy()
            display_df.columns = ["Invoice Number", "Risk Score", "Risk Category", "Anomaly Type", "Explanation"]
            
            # Apply styling
            def highlight_risk(val):
                if val == "High":
                    return "background-color: #FECACA; color: #DC2626; font-weight: bold"
                elif val == "Medium":
                    return "background-color: #FEF3C7; color: #F59E0B; font-weight: bold"
                elif val == "Low":
                    return "background-color: #D1FAE5; color: #10B981; font-weight: bold"
                return ""
            
            # Apply styling to Risk Category column
            styled_df = display_df.style.applymap(highlight_risk, subset=["Risk Category"])
            
            # Format Risk Score as percentage
            styled_df = styled_df.format({"Risk Score": "{:.2%}"})
            
            st.dataframe(styled_df, use_container_width=True)
            
            st.markdown(f"<p>Showing {len(filtered_data)} invoices with risk score ≥ {risk_threshold:.2%}</p>", unsafe_allow_html=True)
        else:
            st.info("No invoices match the selected filters")
        
        # High Risk Invoices
        st.markdown('<h3 class="sub-header">Highest Risk Invoices</h3>', unsafe_allow_html=True)
        
        top_invoices = data["top_risk_invoices"].head(10).copy()
        
        if len(top_invoices) > 0:
            # Format for display
            top_display = top_invoices[["invoice_number", "anomaly_risk_score", "original_anomaly_type", "original_explanation"]].copy()
            top_display.columns = ["Invoice Number", "Risk Score", "Anomaly Type", "Explanation"]
            
            # Format Risk Score as percentage
            top_display["Risk Score"] = top_display["Risk Score"].map("{:.2%}".format)
            
            st.dataframe(top_display, use_container_width=True)
        else:
            st.info("No high risk invoices available")

if __name__ == "__main__":
    main()
