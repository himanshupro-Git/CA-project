import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="AutoML Pipeline Pro", layout="wide")
st.title("🚀 Advanced ML Pipeline Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Step 1: Configuration")
problem_type = st.sidebar.selectbox("Select Problem Type", ["Classification", "Regression"])

# Session State Init
for key in ['df', 'target_col', 'final_features']:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────
# Helper: encode dataframe for ML
# ─────────────────────────────────────────────
def encode_df(df):
    """Return a fully numeric copy of df, encoding dates and categoricals."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(np.int64) // 10**9  # unix timestamp
        elif out[col].dtype == object:
            # Try parsing as date first
            try:
                parsed = pd.to_datetime(out[col])
                out[col] = parsed.astype(np.int64) // 10**9
            except Exception:
                out[col] = LabelEncoder().fit_transform(out[col].astype(str))
    return out

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📥 Data Input",
    "📊 EDA",
    "🛠 Engineering",
    "🎯 Feature Selection",
    "🤖 Training & Tuning"
])

# ══════════════════════════════════════════════
# TAB 1: DATA INPUT + PCA
# ══════════════════════════════════════════════
with tabs[0]:
    uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)

    if st.session_state.df is not None:
        df = st.session_state.df
        st.write(f"**Dataset:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10))

        col1, col2 = st.columns([1, 2])
        with col1:
            target_col = st.selectbox("Select Target Feature", df.columns)
            st.session_state.target_col = target_col

            feature_options = [c for c in df.columns if c != target_col]
            features = st.multiselect(
                "Select Features for PCA/Analysis",
                feature_options,
                default=feature_options[:5]
            )

        with col2:
            if features:
                encoded = encode_df(df[features])
                numeric_df = encoded.dropna()

                if numeric_df.shape[1] >= 2:
                    scaled_data = StandardScaler().fit_transform(numeric_df)
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(scaled_data)
                    color_data = df.loc[numeric_df.index, target_col].astype(str)

                    fig_pca = px.scatter(
                        x=components[:, 0],
                        y=components[:, 1],
                        color=color_data,
                        title=f"PCA 2D Projection (var explained: "
                              f"{pca.explained_variance_ratio_.sum()*100:.1f}%)",
                        labels={'x': 'PC1', 'y': 'PC2'},
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
                else:
                    st.warning("⚠️ Please select at least 2 features for PCA.")

# ══════════════════════════════════════════════
# TAB 2: EDA
# ══════════════════════════════════════════════
with tabs[1]:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.header("Exploratory Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Dataset Summary (numeric)**")
            st.write(df.describe())

            st.write("**Missing Values**")
            st.write(df.isnull().sum().rename("nulls"))

        with col2:
            # Correlation on encoded version
            enc_df = encode_df(df)
            corr = enc_df.corr(numeric_only=True)
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                                 title="Correlation Matrix (encoded)")
            st.plotly_chart(fig_corr, use_container_width=True)

        # Distribution of AQI Value if present
        if "AQI Value" in df.columns:
            st.subheader("AQI Value Distribution")
            fig_hist = px.histogram(df, x="AQI Value", color="Status",
                                    template="plotly_dark", nbins=50)
            st.plotly_chart(fig_hist, use_container_width=True)

        if "Status" in df.columns:
            st.subheader("Status Counts")
            fig_bar = px.bar(df["Status"].value_counts().reset_index(),
                             x="Status", y="count",
                             color="Status", template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3: ENGINEERING
# ══════════════════════════════════════════════
with tabs[2]:
    if st.session_state.df is not None:
        df = st.session_state.df.copy()

        st.subheader("Missing Value Imputation")
        method = st.selectbox("Imputation Method", ["None", "Mean", "Median", "Mode"])

        if method != "None":
            for col in df.select_dtypes(include=[np.number]).columns:
                if method == "Mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif method == "Median":
                    df[col] = df[col].fillna(df[col].median())
                elif method == "Mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
            st.success("✅ Imputation Applied")
            st.session_state.df = df

        st.subheader("Outlier Detection")
        outlier_method = st.selectbox("Method", ["IQR", "Isolation Forest"])

        num_df = encode_df(df).select_dtypes(include=[np.number])

        if not num_df.empty:
            if outlier_method == "IQR":
                Q1 = num_df.quantile(0.25)
                Q3 = num_df.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)
            else:
                iso = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso.fit_predict(num_df) == -1

            st.warning(f"Detected **{int(outliers.sum())}** outliers out of {len(df)} rows")

            if st.button("Remove Outliers"):
                st.session_state.df = df[~outliers.values]
                st.success(f"✅ Removed {int(outliers.sum())} outliers. "
                           f"Remaining rows: {(~outliers).sum()}")
                st.rerun()
        else:
            st.info("No numeric columns detected for outlier analysis.")

# ══════════════════════════════════════════════
# TAB 4: FEATURE SELECTION
# ══════════════════════════════════════════════
with tabs[3]:
    if st.session_state.df is not None and st.session_state.target_col:
        df = st.session_state.df.dropna()
        target_col = st.session_state.target_col

        # Encode everything to numeric
        enc_df = encode_df(df)
        X = enc_df.drop(columns=[target_col])
        y = enc_df[target_col]

        if X.shape[1] == 0:
            st.error("❌ No features available after encoding.")
            st.stop()

        st.write(f"**Features available:** {list(X.columns)}")
        st.write(f"**Target:** `{target_col}` — {y.nunique()} unique values")

        method = st.radio("Selection Criterion", ["Variance Threshold", "Information Gain"])

        if method == "Variance Threshold":
            threshold = st.slider("Threshold", 0.0, float(X.var().max()), 0.0)
            sel = VarianceThreshold(threshold=threshold)
            sel.fit(X)
            selected_features = X.columns[sel.get_support()].tolist()
        else:
            if problem_type == "Classification":
                scores = mutual_info_classif(X, y, random_state=42)
            else:
                scores = mutual_info_regression(X, y, random_state=42)

            feat_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
            fig_fi = px.bar(feat_scores.reset_index(), x="index", y=0,
                            labels={"index": "Feature", "0": "Mutual Info Score"},
                            title="Feature Importance (Mutual Information)",
                            template="plotly_dark")
            st.plotly_chart(fig_fi, use_container_width=True)
            top_n = st.slider("Top N features", 1, len(feat_scores), min(5, len(feat_scores)))
            selected_features = feat_scores.index[:top_n].tolist()

        st.success(f"✅ Selected Features: **{selected_features}**")
        st.session_state.final_features = selected_features

# ══════════════════════════════════════════════
# TAB 5: TRAINING
# ══════════════════════════════════════════════
with tabs[4]:
    if st.session_state.final_features and st.session_state.target_col:
        target_col = st.session_state.target_col
        df = st.session_state.df.dropna()
        enc_df = encode_df(df)

        final_features = [f for f in st.session_state.final_features if f in enc_df.columns]
        if not final_features:
            st.error("❌ Selected features not found in encoded dataset.")
            st.stop()

        X = enc_df[final_features]
        y = enc_df[target_col]

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model_choice = st.selectbox("Model", ["Linear/Logistic Regression", "SVM", "Random Forest"])
        k_fold = st.number_input("K-Fold", 2, 10, 5)

        if model_choice == "Random Forest":
            model = RandomForestClassifier(random_state=42) if problem_type == "Classification" \
                    else RandomForestRegressor(random_state=42)
            params = {"n_estimators": [10, 50, 100]}
        elif model_choice == "SVM":
            model = SVC(random_state=42) if problem_type == "Classification" else SVR()
            params = {"C": [0.1, 1, 10]}
        else:
            model = LogisticRegression(max_iter=1000, random_state=42) if problem_type == "Classification" \
                    else LinearRegression()
            params = {}

        if st.button("🚂 Train Model"):
            with st.spinner("Training..."):
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=k_fold)
                model.fit(X_train_scaled, y_train)
                test_preds = model.predict(X_test_scaled)

            st.success("✅ Training Complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("CV Score (mean)", f"{cv_scores.mean():.4f}")
                st.metric("CV Std", f"{cv_scores.std():.4f}")

            with col2:
                if problem_type == "Classification":
                    st.metric("Test Accuracy", f"{accuracy_score(y_test, test_preds):.4f}")
                    st.text(classification_report(y_test, test_preds))
                else:
                    st.metric("R² Score", f"{r2_score(y_test, test_preds):.4f}")
                    st.metric("MSE", f"{mean_squared_error(y_test, test_preds):.4f}")

            # Actual vs Predicted (regression)
            if problem_type == "Regression":
                fig_pred = px.scatter(x=y_test, y=test_preds,
                                      labels={"x": "Actual", "y": "Predicted"},
                                      title="Actual vs Predicted",
                                      template="plotly_dark")
                fig_pred.add_shape(type="line",
                                   x0=y_test.min(), y0=y_test.min(),
                                   x1=y_test.max(), y1=y_test.max(),
                                   line=dict(color="red", dash="dash"))
                st.plotly_chart(fig_pred, use_container_width=True)

        use_grid = st.checkbox("Run Grid Search (slower)")
        if use_grid and params:
            if st.button("🔍 Run GridSearchCV"):
                with st.spinner("Running grid search..."):
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    grid = GridSearchCV(model, params, cv=3, n_jobs=-1)
                    grid.fit(X_train_scaled, y_train)
                st.success("✅ Grid Search Done!")
                st.write("**Best Params:**", grid.best_params_)
                st.write("**Best Score:**", f"{grid.best_score_:.4f}")
    else:
        st.info("👈 Complete Feature Selection (Tab 4) first.")
