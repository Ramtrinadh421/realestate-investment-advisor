# app.py — Final Streamlit app (complete)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

# -------------------------
# Paths & load checks
# -------------------------
MODEL_DIR = Path("models")
DATA_PATH = Path("data/processed_housing_data.csv")

required_files = {
    "regression": MODEL_DIR / "regression_model.pkl",
    "classification": MODEL_DIR / "classification_model.pkl",
    "encoders": MODEL_DIR / "label_encoders.pkl",
    "scaler": MODEL_DIR / "scaler.pkl",
    "feature_names": MODEL_DIR / "feature_names.pkl",
}

missing = [name for name, path in required_files.items() if not path.exists()]
if missing:
    st.error(f"Missing model/artifact files: {missing}. Put them into the models/ folder.")
    st.stop()

# Load artifacts
reg_model = joblib.load(required_files["regression"])
clf_model = joblib.load(required_files["classification"])
label_encoders = joblib.load(required_files["encoders"])
scaler = joblib.load(required_files["scaler"])
feature_names = joblib.load(required_files["feature_names"])

# Load processed dataset to infer numeric columns (safe)
if DATA_PATH.exists():
    processed_df = pd.read_csv(DATA_PATH)
    numeric_cols_training = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove targets if present
    for t in ("Price_per_SqFt", "Good_Investment"):
        if t in numeric_cols_training:
            numeric_cols_training.remove(t)
else:
    numeric_cols_training = []  # fallback

# -------------------------
# Helper functions
# -------------------------
def get_encoder_classes(label):
    """Return classes for encoder key if exists, else return empty list."""
    return list(label_encoders[label].classes_) if label in label_encoders else []

def safe_transform(label, value):
    """Safely transform a categorical value using the saved LabelEncoder.
    - If encoder missing -> returns 0
    - If value unseen -> add 'Other' to encoder.classes_ and return its code
    """
    if label not in label_encoders:
        return 0
    le = label_encoders[label]
    # ensure value is str for comparison
    val = str(value)
    if val in le.classes_:
        return int(le.transform([val])[0])
    else:
        # add "Other" class dynamically if not present
        classes = list(le.classes_)
        if "Other" not in classes:
            classes.append("Other")
            le.classes_ = np.array(classes)
        return int(le.transform(["Other"])[0])

def build_input_row(user_inputs: dict):
    """Create a DataFrame row aligned with feature_names and encoded/scaled where needed."""
    # Start with zeros for all features
    row = {col: 0 for col in feature_names}

    # Map user inputs into row (categorical or numeric)
    # Numeric fields we expect user to supply:
    numeric_map = {
        "Size_in_SqFt": user_inputs.get("Size_in_SqFt"),
        "BHK": user_inputs.get("BHK"),
        "Year_Built": user_inputs.get("Year_Built"),
        "Age_of_Property": user_inputs.get("Age_of_Property"),
        "Nearby_Schools": user_inputs.get("Nearby_Schools"),
        "Nearby_Hospitals": user_inputs.get("Nearby_Hospitals"),
    }

    for k, v in numeric_map.items():
        if k in row and v is not None:
            row[k] = v

    # Categorical mapping (if encoder exists)
    categorical_labels = [
        "State", "City", "Locality", "Property_Type", "Furnished_Status",
        "Floor_No", "Total_Floors", "Public_Transport_Accessibility",
        "Parking_Space", "Security", "Amenities", "Facing", "Owner_Type", "Availability_Status"
    ]
    for cat in categorical_labels:
        if cat in row:
            supplied = user_inputs.get(cat)
            if supplied is None:
                # leave default 0
                continue
            # For numeric categories (e.g. Floor_No or Total_Floors) if user passed int—store directly
            if isinstance(supplied, (int, float)) and cat in ("Floor_No", "Total_Floors"):
                row[cat] = supplied
            else:
                row[cat] = safe_transform(cat, supplied)

    # Ensure types: numeric columns should be numeric
    for c in numeric_cols_training:
        if c in row:
            try:
                row[c] = float(row[c])
            except Exception:
                # if cannot cast, set 0
                row[c] = 0.0

    # Create DataFrame and align columns in the feature_names order
    df_row = pd.DataFrame([row], columns=feature_names)
    return df_row

def scale_numeric_columns(df_row):
    """Scale numeric columns using scaler; scaler was fit on numeric columns (not targets)."""
    # If scaler has feature_names_in_ attribute, use it; else use numeric_cols_training
    try:
        scaler_cols = list(getattr(scaler, "feature_names_in_", []))
        if not scaler_cols:
            scaler_cols = numeric_cols_training
    except Exception:
        scaler_cols = numeric_cols_training

    # If scaler_cols empty, attempt to infer numeric columns from df_row
    if not scaler_cols:
        scaler_cols = [c for c in df_row.columns if np.issubdtype(df_row[c].dtype, np.number)]

    # Only scale columns that exist in df_row
    scaler_cols = [c for c in scaler_cols if c in df_row.columns]
    if scaler_cols:
        df_row_scaled = df_row.copy()
        df_row_scaled[scaler_cols] = scaler.transform(df_row_scaled[scaler_cols])
        return df_row_scaled
    return df_row

# -------------------------
# Build UI
# -------------------------
st.title("🏠 Real Estate Investment Advisor")
st.write("Use the form below to enter property details and get a price & investment recommendation.")

with st.form("property_form"):
    st.subheader("Property details")

    # Left column inputs (numeric)
    c1, c2 = st.columns(2)
    with c1:
        State = st.selectbox("State", get_encoder_classes("State") or ["Unknown"])
        City = st.selectbox("City", get_encoder_classes("City") or ["Unknown"])
        Locality = st.selectbox("Locality", get_encoder_classes("Locality") or ["Unknown"])
        Property_Type = st.selectbox("Property Type", get_encoder_classes("Property_Type") or ["Apartment"])
        BHK = st.number_input("BHK", min_value=1, max_value=10, value=2)
        Size_in_SqFt = st.number_input("Size (SqFt)", min_value=200, max_value=20000, value=1200)
        Year_Built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2018)

    with c2:
        Furnished_Status = st.selectbox("Furnished Status", get_encoder_classes("Furnished_Status") or ["Unfurnished"])
        Floor_No = st.number_input("Floor No", min_value=0, max_value=200, value=1)
        Total_Floors = st.number_input("Total Floors", min_value=1, max_value=200, value=10)
        Age_of_Property = st.number_input("Age of Property", min_value=0, max_value=200, value=5)
        Nearby_Schools = st.number_input("Nearby Schools (count/rating)", min_value=0, max_value=50, value=3)
        Nearby_Hospitals = st.number_input("Nearby Hospitals (count/rating)", min_value=0, max_value=50, value=2)
        Public_Transport_Accessibility = st.selectbox("Public Transport Accessibility", get_encoder_classes("Public_Transport_Accessibility") or ["Low", "Medium", "High"])

    # bottom row categorical
    Amenities = st.selectbox("Amenities", get_encoder_classes("Amenities") or ["None"])
    Parking_Space = st.selectbox("Parking Space", get_encoder_classes("Parking_Space") or ["No Parking"])
    Security = st.selectbox("Security", get_encoder_classes("Security") or ["Low", "Medium", "High"])
    Facing = st.selectbox("Facing", get_encoder_classes("Facing") or ["North", "South", "East", "West"])
    Owner_Type = st.selectbox("Owner Type", get_encoder_classes("Owner_Type") or ["Individual"])
    Availability_Status = st.selectbox("Availability Status", get_encoder_classes("Availability_Status") or ["Available"])

    submitted = st.form_submit_button("Predict")

# -------------------------
# When submit, build input, scale and predict
# -------------------------
if submitted:
    user_inputs = {
        "State": State,
        "City": City,
        "Locality": Locality,
        "Property_Type": Property_Type,
        "BHK": BHK,
        "Size_in_SqFt": Size_in_SqFt,
        "Year_Built": Year_Built,
        "Furnished_Status": Furnished_Status,
        "Floor_No": Floor_No,
        "Total_Floors": Total_Floors,
        "Age_of_Property": Age_of_Property,
        "Nearby_Schools": Nearby_Schools,
        "Nearby_Hospitals": Nearby_Hospitals,
        "Public_Transport_Accessibility": Public_Transport_Accessibility,
        "Amenities": Amenities,
        "Parking_Space": Parking_Space,
        "Security": Security,
        "Facing": Facing,
        "Owner_Type": Owner_Type,
        "Availability_Status": Availability_Status,
    }

    # Build input DataFrame aligned with feature_names
    input_df = build_input_row(user_inputs)

    # Scale numeric columns using scaler
    input_scaled = scale_numeric_columns(input_df)

    # Predict regression & classification
    try:
        pred_price_per_sqft = reg_model.predict(input_scaled)[0]
    except Exception as e:
        st.error(f"Regression prediction failed: {e}")
        st.stop()

    try:
        pred_invest = clf_model.predict(input_scaled)[0]
    except Exception as e:
        st.error(f"Classification prediction failed: {e}")
        st.stop()

    # If predicted price is unrealistically small (<1), assume model predicted a scaled value -> scale heuristically
    if pred_price_per_sqft < 1:
        # heuristic: multiply by 10000 (this was used earlier in conversation). Adjust if your training target was different.
        pred_price_per_sqft_display = float(pred_price_per_sqft) * 10000
    else:
        pred_price_per_sqft_display = float(pred_price_per_sqft)

    total_price = pred_price_per_sqft_display * Size_in_SqFt

    # -------------------------
    # Display results
    # -------------------------
    st.subheader("🎯 Prediction Results")
    colA, colB = st.columns(2)
    with colA:
        st.metric("Predicted Price per SqFt", f"₹ {pred_price_per_sqft_display:,.2f}")
        st.metric("Estimated Total Property Price", f"₹ {total_price:,.0f}")
    with colB:
        if int(pred_invest) == 1:
            st.success("👍 Good Investment")
        else:
            st.error("👎 Not Recommended")

    # Investment score (0-100) — use classification probability when possible
    try:
        prob = clf_model.predict_proba(input_scaled)[0]
        score = float(prob[1] * 100)  # probability of being Good_Investment
    except Exception:
        # fallback: derive score from price relative to range
        score = min(max((pred_price_per_sqft_display - 1000) / (25000 - 1000) * 100, 0), 100)

    # Score meter (simple horizontal bar plot)
    st.subheader("📊 Investment Score")
    fig, ax = plt.subplots(figsize=(6, 0.8))
    ax.barh([0], [score], color="green" if score >= 50 else "orange")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Score (%)")
    st.pyplot(fig)

    # 5-Year appreciation projection chart (simple linear growth at 6% p.a.)
    st.subheader("📈 5-Year Value Projection (6% p.a.)")
    years = list(range(2025, 2031))
    projection = [total_price * ((1 + 0.06) ** i) for i in range(6)]
    fig2, ax2 = plt.subplots()
    ax2.plot(years, projection, marker="o")
    ax2.set_ylabel("Estimated Property Value (₹)")
    ax2.set_xlabel("Year")
    ax2.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig2)

    # Investment tips
    st.subheader("💡 Investment Tips")
    if score >= 70:
        st.success("High score — strong buy candidate. Consider proceeding after inspection.")
    elif score >= 50:
        st.info("Moderate score — consider negotiation and local comps.")
    else:
        st.warning("Low score — consider alternate properties or negotiation.")

    # quick textual tips
    st.markdown("""
    **Tips**
    - Check RERA, ownership documents, and builder reputation.  
    - Compare with 2–3 nearby listings before buying.  
    - Consider rental yield if planning to rent out.  
    """)

