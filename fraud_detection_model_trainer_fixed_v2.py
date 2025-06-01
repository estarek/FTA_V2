#!/usr/bin/env python3
"""
E-Invoice Fraud Detection Model Trainer

This script trains a fraud detection model on synthetic e-invoice data and saves
the model and all necessary preprocessors for use in a Streamlit application.

FIXED VERSION: Removed 'anomaly_type' from features to prevent data leakage.
Fixed feature importance mismatch issue.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import os
import joblib
import json
import pickle

# --- Configuration ---
DATA_DIR = "output"
INVOICE_FILE = os.path.join(DATA_DIR, "invoices.csv")
TAXPAYER_FILE = os.path.join(DATA_DIR, "taxpayers.csv")
MODEL_DIR = "model_artifacts"
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Create model artifacts directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Load and Preprocess Data ---
def load_and_preprocess_data(invoice_path, taxpayer_path):
    """Loads invoice and taxpayer data, merges them, and performs initial preprocessing."""
    print(f"Loading data from {invoice_path} and {taxpayer_path}...")
    try:
        # Add on_bad_lines="skip" to handle potential parsing errors in invoices.csv
        print(f"Attempting to load {invoice_path} with error skipping...")
        invoices_df = pd.read_csv(invoice_path, low_memory=False, on_bad_lines="skip")
        print("Invoice loading complete. Note: Bad lines may have been skipped.")
        
        taxpayers_df = pd.read_csv(taxpayer_path, low_memory=False)
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure the CSV files are in the {DATA_DIR} directory.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during CSV loading: {e}")
        return None

    print(f"Loaded {len(invoices_df)} invoices and {len(taxpayers_df)} taxpayers.")

    # --- Data Cleaning & Type Conversion ---
    # Convert boolean columns properly (handle potential string representations)
    invoices_df["is_anomaly"] = invoices_df["is_anomaly"].astype(str).str.lower() == "true"
    taxpayers_df["is_anomaly"] = taxpayers_df["is_anomaly"].astype(str).str.lower() == "true"

    # Convert datetime columns
    invoices_df["invoice_datetime"] = pd.to_datetime(invoices_df["invoice_datetime"], errors="coerce")
    taxpayers_df["registration_date"] = pd.to_datetime(taxpayers_df["registration_date"], errors="coerce")
    taxpayers_df["vat_registration_date"] = pd.to_datetime(taxpayers_df["vat_registration_date"], errors="coerce")

    # Handle potential missing values (fill numeric with 0 or median, categorical with "UNKNOWN")
    numeric_cols_inv = invoices_df.select_dtypes(include=np.number).columns
    invoices_df[numeric_cols_inv] = invoices_df[numeric_cols_inv].fillna(0)
    
    numeric_cols_tax = taxpayers_df.select_dtypes(include=np.number).columns
    taxpayers_df[numeric_cols_tax] = taxpayers_df[numeric_cols_tax].fillna(0)

    categorical_cols_inv = invoices_df.select_dtypes(include="object").columns
    invoices_df[categorical_cols_inv] = invoices_df[categorical_cols_inv].fillna("UNKNOWN")
    
    categorical_cols_tax = taxpayers_df.select_dtypes(include="object").columns
    taxpayers_df[categorical_cols_tax] = taxpayers_df[categorical_cols_tax].fillna("UNKNOWN")

    # --- Merge Data ---
    # Rename taxpayer columns to avoid conflicts before merging
    taxpayers_df = taxpayers_df.add_prefix("seller_")
    
    # Merge invoice data with seller (taxpayer) data
    # Use seller_trn from invoices which matches seller_tax_number from taxpayers
    merged_df = pd.merge(
        invoices_df,
        taxpayers_df,
        left_on="seller_trn",
        right_on="seller_tax_number",
        how="left",
        suffixes=("_inv", "_tax")
    )
    
    # Fill any NaNs introduced by the merge
    numeric_cols_merged = merged_df.select_dtypes(include=np.number).columns
    merged_df[numeric_cols_merged] = merged_df[numeric_cols_merged].fillna(0)
    categorical_cols_merged = merged_df.select_dtypes(include="object").columns
    merged_df[categorical_cols_merged] = merged_df[categorical_cols_merged].fillna("UNKNOWN")

    print(f"Data merged. Resulting shape: {merged_df.shape}")
    
    # Drop redundant columns after merge if necessary
    if "seller_tax_number" in merged_df.columns:
         merged_df.drop(columns=["seller_tax_number"], inplace=True)

    return merged_df

# --- 2. Feature Engineering ---
def engineer_features(df):
    """Engineers features for the fraud detection model."""
    print("\n--- Starting Feature Engineering --- ")
    
    # Target variable
    y = df["is_anomaly"].astype(int)
    
    # --- Feature Selection ---
    # Select a subset of potentially relevant features
    # FIXED: Removed 'anomaly_type' from features to prevent data leakage
    features = [
        # Invoice numerical features
        "invoice_discount_amount", "invoice_without_tax", "invoice_tax_amount",
        "vat_rate", "taxable_amount", "emirate_revenue_share",
        # Invoice categorical features
        "invoice_type", "invoice_category", "invoice_sales_type", "invoice_collection_type",
        "document_status", "buyer_emirate", "buyer_sector", "vat_category", 
        "submission_channel", "geo_boundary_type",
        # Invoice time features
        "invoice_year", "invoice_month", "invoice_quarter", "invoice_week", "invoice_day_of_week",
        # Seller numerical features (from merged taxpayer data)
        "seller_number_of_employees", "seller_tax_compliance_score",
        # Seller categorical features
        "seller_sector", "seller_business_size", "seller_legal_entity_type", "seller_ownership_type",
        "seller_bank_country"
        # REMOVED: "anomaly_type" - This would cause data leakage
    ]
    
    # Add time-based features if datetime columns are valid
    if pd.api.types.is_datetime64_any_dtype(df["invoice_datetime"]):
        df["invoice_hour"] = df["invoice_datetime"].dt.hour
        features.append("invoice_hour")
        
    if (pd.api.types.is_datetime64_any_dtype(df["invoice_datetime"]) and
        pd.api.types.is_datetime64_any_dtype(df["seller_registration_date"])):
        # Calculate time since seller registration in days
        time_diff = (df["invoice_datetime"] - df["seller_registration_date"]).dt.days
        df["days_since_seller_reg"] = time_diff.fillna(-1) # Fill NaT differences with -1
        features.append("days_since_seller_reg")

    # --- Feature Creation ---
    # Example: Ratio of tax to taxable amount
    df["tax_ratio"] = np.where(df["taxable_amount"] != 0, 
                               df["invoice_tax_amount"] / df["taxable_amount"], 
                               0)
    features.append("tax_ratio")

    # Select the final feature set
    # Ensure only features present in the dataframe are selected
    valid_features = [f for f in features if f in df.columns]
    
    # ADDED: Extra check to ensure anomaly_type is not in features
    if "anomaly_type" in valid_features:
        valid_features.remove("anomaly_type")
        print("WARNING: 'anomaly_type' was found in features and has been removed to prevent data leakage.")
    
    X = df[valid_features].copy()
    print(f"Selected {len(valid_features)} features: {valid_features}")

    # --- Encoding Categorical Features ---
    categorical_features = X.select_dtypes(include="object").columns
    print(f"Encoding categorical features: {categorical_features.tolist()}")
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        # Fit on the entire column in the original df to handle unseen values if splitting later
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le # Store encoder for later use
        
    # --- Scaling Numerical Features ---
    numerical_features = X.select_dtypes(include=np.number).columns
    print(f"Scaling numerical features: {numerical_features.tolist()}")
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    print("Feature engineering complete.")
    
    # Save feature metadata for Streamlit app
    feature_metadata = {
        "categorical_features": categorical_features.tolist(),
        "numerical_features": numerical_features.tolist(),
        "valid_features": valid_features
    }
    
    return X, y, scaler, label_encoders, feature_metadata

# --- 3. Train and Evaluate Model ---
def train_and_evaluate(X, y):
    """Splits data, trains a RandomForestClassifier, and evaluates its performance."""
    print("\n--- Starting Model Training and Evaluation --- ")
    
    # Split data into training and testing sets (stratified due to potential class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Anomaly distribution in training set:\n{y_train.value_counts(normalize=True)}")
    print(f"Anomaly distribution in test set:\n{y_test.value_counts(normalize=True)}")

    # Initialize the classifier
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    
    # Train the model
    print("Training RandomForestClassifier model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Make predictions on the test set
    print("Making predictions on the test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class (anomaly)

    # Evaluate the model
    print("\n--- Model Evaluation Results --- ")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=["Not Anomaly", "Anomaly"], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=["Not Anomaly", "Anomaly"]))
    
    # Create evaluation metrics dictionary for Streamlit app
    evaluation_metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "test_indices": X_test.index.tolist(),
        "y_test": y_test.tolist(),
        "y_pred_proba": y_pred_proba.tolist()
    }
    
    return model, X_test, y_test, y_pred_proba, evaluation_metrics

# --- 4. Save Sample Results for Streamlit ---
def save_sample_results(df, X_test, y_test, y_pred_proba):
    """Saves sample results for the Streamlit app."""
    print("\n--- Saving Sample Results for Streamlit --- ")
    
    # Get original indices from X_test to link back to the original dataframe
    test_indices = X_test.index
    
    # Create a results DataFrame
    results_df = pd.DataFrame({
        "invoice_number": df.loc[test_indices, "invoice_number"],
        "true_anomaly": y_test,
        "anomaly_risk_score": y_pred_proba,
        "predicted_anomaly": (y_pred_proba >= 0.5).astype(int),
        # Keep anomaly_type and explanation for display purposes only, not for model training
        "original_anomaly_type": df.loc[test_indices, "anomaly_type"],
        "original_explanation": df.loc[test_indices, "anomaly_explanation"]
    })
    
    # Sort by risk score (descending)
    results_df_sorted = results_df.sort_values(by="anomaly_risk_score", ascending=False)
    
    # Save top and bottom 50 for Streamlit app
    top_invoices = results_df_sorted.head(50)
    bottom_invoices = results_df_sorted.tail(50)
    
    # Save the full results dataframe for Streamlit app
    results_df_sorted.to_csv(os.path.join(MODEL_DIR, "risk_scored_invoices.csv"), index=False)
    
    # Save sample invoices for quick display
    top_invoices.to_csv(os.path.join(MODEL_DIR, "top_risk_invoices.csv"), index=False)
    bottom_invoices.to_csv(os.path.join(MODEL_DIR, "bottom_risk_invoices.csv"), index=False)
    
    # Get anomaly type distribution for visualization
    anomaly_type_counts = df[df["is_anomaly"] == True]["anomaly_type"].value_counts().reset_index()
    anomaly_type_counts.columns = ["anomaly_type", "count"]
    anomaly_type_counts.to_csv(os.path.join(MODEL_DIR, "anomaly_type_distribution.csv"), index=False)
    
    # Get emirate distribution for visualization
    emirate_counts = df["buyer_emirate"].value_counts().reset_index()
    emirate_counts.columns = ["emirate", "count"]
    emirate_counts.to_csv(os.path.join(MODEL_DIR, "emirate_distribution.csv"), index=False)
    
    print(f"Sample results saved to {MODEL_DIR} directory.")
    
    return results_df_sorted

# --- Main Execution Flow ---
if __name__ == "__main__":
    print("--- Starting Fraud Detection Model Training (FIXED VERSION) --- ")
    print("NOTE: 'anomaly_type' has been removed from features to prevent data leakage")
    print("NOTE: Fixed feature importance mismatch issue")
    
    # Step 1: Load and Preprocess
    df = load_and_preprocess_data(INVOICE_FILE, TAXPAYER_FILE)
    
    if df is not None:
        print("\n--- Data Loading and Preprocessing Complete ---")
        print(f"Target variable distribution (is_anomaly):\n{df['is_anomaly'].value_counts(normalize=True)}")
        
        # Step 2: Feature Engineering
        X, y, scaler, label_encoders, feature_metadata = engineer_features(df)
        
        print("\n--- Feature Engineering Complete ---")
        print(f"Shape of feature matrix X: {X.shape}")
        print(f"Shape of target vector y: {y.shape}")

        # Step 3: Train and Evaluate Model
        model, X_test, y_test, y_pred_proba, evaluation_metrics = train_and_evaluate(X, y)

        # Step 4: Save Sample Results for Streamlit
        results_df = save_sample_results(df, X_test, y_test, y_pred_proba)
        
        # Step 5: Save Model and Preprocessors for Streamlit
        print("\n--- Saving Model and Preprocessors for Streamlit --- ")
        
        # Save the model
        joblib.dump(model, os.path.join(MODEL_DIR, "fraud_detection_model.joblib"))
        
        # Save the scaler
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
        
        # Save the label encoders
        with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
            pickle.dump(label_encoders, f)
        
        # FIXED: Ensure feature_metadata matches model's feature_importances_
        # Extract feature importances from the model
        feature_importances = model.feature_importances_
        
        # Verify and fix any mismatch between feature_importances and feature_metadata
        if len(feature_importances) != len(feature_metadata["valid_features"]):
            print(f"WARNING: Feature importance length ({len(feature_importances)}) doesn't match feature names length ({len(feature_metadata['valid_features'])})")
            print("Fixing feature metadata to match model's feature importances...")
            
            # Get the actual features used by the model from X
            actual_features = X.columns.tolist()
            
            # Update feature_metadata with the actual features used
            feature_metadata["valid_features"] = actual_features
            
            # Recategorize features
            feature_metadata["categorical_features"] = [f for f in actual_features if f in feature_metadata["categorical_features"]]
            feature_metadata["numerical_features"] = [f for f in actual_features if f in feature_metadata["numerical_features"] or f not in feature_metadata["categorical_features"]]
            
            print(f"Fixed feature metadata now has {len(feature_metadata['valid_features'])} features")
        
        # Verify the fix worked
        print(f"Final check - Feature importance length: {len(feature_importances)}, Feature names length: {len(feature_metadata['valid_features'])}")
        
        # Save feature metadata
        with open(os.path.join(MODEL_DIR, "feature_metadata.json"), "w") as f:
            json.dump(feature_metadata, f)
        
        # Save evaluation metrics
        with open(os.path.join(MODEL_DIR, "evaluation_metrics.json"), "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(evaluation_metrics, f)
        
        print(f"Model and preprocessors saved to {MODEL_DIR} directory.")
        print("\n--- Model Training Complete --- ")

    else:
        print("Failed to load data. Exiting.")

    print("\n--- Fraud Detection Model Training Script End --- ")
