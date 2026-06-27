"""
ML Visualizer Lab — Preprocessing
Feature/target selection, missing-value handling, encoding, scaling, train-test split.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.model_selection import train_test_split

# ─── Guard ─────────────────────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("df")
if df is None:
    st.warning("⚠️ No dataset loaded. Go to **Dataset Overview** first.")
    st.stop()

st.markdown("## 🔧 Data Preprocessing")
st.caption(f"Dataset: *{st.session_state.get('dataset_name', '')}* — {df.shape[0]} × {df.shape[1]}")

# ─── 1. Feature & Target Selection ────────────────────────────────────────────
st.markdown("### 1️⃣ Feature & Target Selection")

col_left, col_right = st.columns([3, 1])

with col_right:
    target_col = st.selectbox(
        "Target column",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,
        help="The column your model will predict.",
    )

with col_left:
    default_features = [
        c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    feature_cols = st.multiselect(
        "Feature columns",
        options=[c for c in df.columns if c != target_col],
        default=default_features,
        help="Select one or more columns to use as features.",
    )

# Auto-detect task type
_target_nunique = df[target_col].nunique()
_target_is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
try:
    _is_integer = df[target_col].dropna().apply(float.is_integer).all() if _target_is_numeric else False
except (TypeError, ValueError):
    _is_integer = False

if (_target_nunique <= 20 and _target_is_numeric and _is_integer) or not _target_is_numeric:
    detected_task = "classification"
else:
    detected_task = "regression"

st.info(f"🔍 Detected task type: **{detected_task}** ({_target_nunique} unique target values)")

if not feature_cols:
    st.warning("Select at least one feature column.")
    st.stop()

st.markdown("---")

# ─── 2. Missing Values ────────────────────────────────────────────────────────
st.markdown("### 2️⃣ Missing Values")

selected_df = df[feature_cols + [target_col]].copy()
missing_total = selected_df.isnull().sum()
missing_any = missing_total[missing_total > 0]

if missing_any.empty:
    st.success("✅ No missing values in selected columns.")
    missing_strategy = "None"
else:
    st.dataframe(
        pd.DataFrame({"Column": missing_any.index, "Missing": missing_any.values, "% Missing": (missing_any.values / len(selected_df) * 100).round(1)}),
        hide_index=True,
        use_container_width=True,
    )
    missing_strategy = st.selectbox(
        "Strategy",
        ["Drop rows", "Mean", "Median", "Most Frequent"],
        help="How to handle missing values.",
    )

st.markdown("---")

# ─── 3. Categorical Encoding ──────────────────────────────────────────────────
st.markdown("### 3️⃣ Categorical Encoding")

cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

if not cat_cols:
    st.success("✅ No categorical feature columns detected.")
    encoding_method = "None"
else:
    st.write(f"Categorical columns: **{', '.join(cat_cols)}**")
    encoding_method = st.selectbox(
        "Encoding method",
        ["Label Encoding", "One-Hot Encoding"],
    )

st.markdown("---")

# ─── 4. Feature Scaling ───────────────────────────────────────────────────────
st.markdown("### 4️⃣ Feature Scaling")

scaling_option = st.selectbox(
    "Scaler",
    ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"],
)

st.markdown("---")

# ─── 5. Train-Test Split ──────────────────────────────────────────────────────
st.markdown("### 5️⃣ Train–Test Split")

train_pct = st.slider("Training set size (%)", min_value=50, max_value=90, value=80, step=5)

st.markdown("---")

# ─── 6. Apply ─────────────────────────────────────────────────────────────────
apply_btn = st.button("✅ Apply Preprocessing", type="primary", use_container_width=True)

if apply_btn:
    try:
        with st.spinner("Preprocessing…"):
            proc = selected_df.copy()

            # --- Missing values ---------------------------------------------------
            if missing_strategy == "Drop rows":
                proc.dropna(inplace=True)
            elif missing_strategy == "Mean":
                for c in proc.columns:
                    if proc[c].isnull().any() and pd.api.types.is_numeric_dtype(proc[c]):
                        proc[c].fillna(proc[c].mean(), inplace=True)
            elif missing_strategy == "Median":
                for c in proc.columns:
                    if proc[c].isnull().any() and pd.api.types.is_numeric_dtype(proc[c]):
                        proc[c].fillna(proc[c].median(), inplace=True)
            elif missing_strategy == "Most Frequent":
                for c in proc.columns:
                    if proc[c].isnull().any():
                        proc[c].fillna(proc[c].mode().iloc[0], inplace=True)

            # Fill remaining non-numeric NaNs with mode, then drop any leftover
            for c in proc.columns:
                if proc[c].isnull().any() and not pd.api.types.is_numeric_dtype(proc[c]):
                    if not proc[c].mode().empty:
                        proc[c].fillna(proc[c].mode().iloc[0], inplace=True)
            proc.dropna(inplace=True)

            if len(proc) == 0:
                st.error("❌ No rows remain after handling missing values. Try a different strategy or check your data.")
                st.stop()

            if len(proc) < 10:
                st.warning(f"⚠️ Only **{len(proc)} rows** remain after handling missing values. Results may be unreliable.")

            # --- Encode target if needed ------------------------------------------
            if not pd.api.types.is_numeric_dtype(proc[target_col]):
                le_target = LabelEncoder()
                proc[target_col] = le_target.fit_transform(proc[target_col].astype(str))

            # --- Categorical encoding ---------------------------------------------
            actual_cat = [c for c in cat_cols if c in proc.columns]
            if actual_cat and encoding_method == "Label Encoding":
                for c in actual_cat:
                    le = LabelEncoder()
                    proc[c] = le.fit_transform(proc[c].astype(str))
            elif actual_cat and encoding_method == "One-Hot Encoding":
                before_cols = len(proc.columns)
                proc = pd.get_dummies(proc, columns=actual_cat, drop_first=True)
                # Convert bool columns to int
                for c in proc.columns:
                    if proc[c].dtype == "bool":
                        proc[c] = proc[c].astype(int)
                after_cols = len(proc.columns)
                if after_cols > before_cols + 50:
                    st.warning(f"⚠️ One-hot encoding created {after_cols - before_cols} new columns. Consider using Label Encoding for high-cardinality features.")

            # --- Separate X and y -------------------------------------------------
            y = proc[target_col].values
            X = proc.drop(columns=[target_col]).values
            final_feature_names = [c for c in proc.columns if c != target_col]

            if X.shape[1] == 0:
                st.error("❌ No features remain after preprocessing. Please select valid feature columns.")
                st.stop()

            # --- Scaling ----------------------------------------------------------
            scaler = None
            if scaling_option == "StandardScaler":
                scaler = StandardScaler()
            elif scaling_option == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif scaling_option == "RobustScaler":
                scaler = RobustScaler()

            if scaler is not None:
                X = scaler.fit_transform(X)

            # --- Split ------------------------------------------------------------
            test_fraction = 1 - (train_pct / 100)

            # Stratification: only for classification, and only if all classes
            # have enough samples for the split
            stratify_target = None
            if detected_task == "classification":
                from collections import Counter
                class_counts = Counter(y)
                min_class_count = min(class_counts.values())
                n_test = max(1, int(len(y) * test_fraction))
                n_train = len(y) - n_test
                n_classes = len(class_counts)
                if min_class_count >= 2 and n_test >= n_classes and n_train >= n_classes:
                    stratify_target = y
                else:
                    st.info("ℹ️ Some classes have too few samples for stratified splitting. Using random split instead.")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_fraction, random_state=42, stratify=stratify_target
            )

            # --- Store in session state -------------------------------------------
            st.session_state["df_processed"] = proc
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            st.session_state["feature_names"] = final_feature_names
            st.session_state["target_name"] = target_col
            st.session_state["task_type"] = detected_task
            st.session_state["scaler"] = scaler

        st.success(
            f"✅ Preprocessing complete!  \n"
            f"**X_train**: {X_train.shape} · **X_test**: {X_test.shape}  \n"
            f"**Task**: {detected_task} · **Features**: {len(final_feature_names)}"
        )

        with st.expander("Preview preprocessed data", expanded=False):
            preview = pd.DataFrame(X_train[:20], columns=final_feature_names)
            preview[target_col] = y_train[:20]
            st.dataframe(preview, use_container_width=True)

    except Exception as e:
        st.error(
            f"❌ **Preprocessing failed**: {str(e)}\n\n"
            "**Suggestions:**\n"
            "- Check that your selected features contain valid data\n"
            "- Try a different missing value strategy\n"
            "- Ensure the target column has consistent values\n"
            "- Try Label Encoding instead of One-Hot for high-cardinality columns"
        )
