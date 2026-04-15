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

# Page Config
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
# Helper: encode dataframe — 100% numeric output
# ─────────────────────────────────────────────
def encode_df(df):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        try:
            parsed = pd.to_datetime(out[col], infer_datetime_format=True)
            out[col] = parsed.astype('int64') // 10**9
            continue
        except Exception:
            pass
        out[col] = LabelEncoder().fit_transform(out[col].astype(str))
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = LabelEncoder().fit_transform(out[col].astype(str))
    return out.astype(float)

# Tabs
tabs = st.tabs([
    "📥 Data Input",
    "📊 EDA",
    "🛠 Engineering",
    "🎯 Feature Selection",
    "🤖 Training & Tuning"
])

# TAB 1
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
            default_index = list(df.columns).index("AQI Value") if "AQI Value" in df.columns else 0

            target_col = st.selectbox(
                "Select Target Feature",
                df.columns,
                index=default_index
            )
            st.session_state.target_col = target_col
            feature_options = [c for c in df.columns if c != target_col]
            features = st.multiselect(
                "Select Features for PCA/Analysis",
                feature_options,
                default=feature_options[:min(5, len(feature_options))]
            )

        with col2:
            if features:
                try:
                    numeric_df = encode_df(df[features]).dropna()
                    if numeric_df.shape[1] >= 2:
                        scaled_data = StandardScaler().fit_transform(numeric_df)
                        pca = PCA(n_components=2)
                        components = pca.fit_transform(scaled_data)
                        color_data = df.loc[numeric_df.index, target_col].astype(str)
                        fig_pca = px.scatter(
                            x=components[:, 0], y=components[:, 1],
                            color=color_data,
                            title=f"PCA 2D Projection (var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
                            labels={'x': 'PC1', 'y': 'PC2'},
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)
                    else:
                        st.warning("⚠️ Please select at least 2 features for PCA.")
                except Exception as e:
                    st.error(f"PCA error: {e}")

# TAB 2
with tabs[1]:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.header("Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Summary**")
            st.write(df.describe())
            st.write("**Missing Values**")
            st.write(df.isnull().sum().rename("nulls"))
        with col2:
            try:
                enc_df = encode_df(df)
                corr = enc_df.corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.error(f"Correlation error: {e}")

        if "AQI Value" in df.columns and "Status" in df.columns:
            st.subheader("AQI Value Distribution")
            fig_hist = px.histogram(df, x="AQI Value", color="Status", template="plotly_dark", nbins=50)
            st.plotly_chart(fig_hist, use_container_width=True)

        if "Status" in df.columns:
            st.subheader("Status Counts")
            vc = df["Status"].value_counts().reset_index()
            vc.columns = ["Status", "count"]
            fig_bar = px.bar(vc, x="Status", y="count", color="Status", template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)

# TAB 3
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
        try:
            num_df = encode_df(df).dropna()
            if not num_df.empty:
                if outlier_method == "IQR":
                    Q1 = num_df.quantile(0.25)
                    Q3 = num_df.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)
                else:
                    iso = IsolationForest(contamination=0.1, random_state=42)
                    outliers = pd.Series(iso.fit_predict(num_df) == -1, index=num_df.index)
                st.warning(f"Detected **{int(outliers.sum())}** outliers out of {len(df)} rows")
                if st.button("Remove Outliers"):
                    st.session_state.df = df.loc[~outliers]
                    st.success(f"✅ Removed outliers. Remaining rows: {(~outliers).sum()}")
                    st.rerun()
        except Exception as e:
            st.error(f"Outlier detection error: {e}")

# TAB 4
with tabs[3]:
    if st.session_state.df is not None and st.session_state.target_col:
        df = st.session_state.df.dropna()
        target_col = st.session_state.target_col
        try:
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
                                title="Feature Importance", template="plotly_dark")
                st.plotly_chart(fig_fi, use_container_width=True)
                top_n = st.slider("Top N features", 1, len(feat_scores), min(5, len(feat_scores)))
                selected_features = feat_scores.index[:top_n].tolist()

            st.success(f"✅ Selected Features: **{selected_features}**")
            st.session_state.final_features = selected_features
        except Exception as e:
            st.error(f"Feature selection error: {e}")

# TAB 5
with tabs[4]:
    if st.session_state.final_features and st.session_state.target_col:
        target_col = st.session_state.target_col
        df = st.session_state.df.dropna()
        try:
            enc_df = encode_df(df)
            final_features = [f for f in st.session_state.final_features if f in enc_df.columns]
            if not final_features:
                st.error("❌ Selected features not found. Go back to Feature Selection tab.")
                st.stop()

            X = enc_df[final_features]
            y = enc_df[target_col]

            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model_choice = st.selectbox("Model", ["Linear/Logistic Regression", "SVM", "Random Forest"])
            k_fold = st.number_input("K-Fold", 2, 10, 5)

            if model_choice == "Random Forest":
                model = RandomForestClassifier(random_state=42) if problem_type == "Classification" else RandomForestRegressor(random_state=42)
                params = {"n_estimators": [10, 50, 100]}
            elif model_choice == "SVM":
                model = SVC(random_state=42) if problem_type == "Classification" else SVR()
                params = {"C": [0.1, 1, 10]}
            else:
                model = LogisticRegression(max_iter=1000, random_state=42) if problem_type == "Classification" else LinearRegression()
                params = {}

            if st.button("🚂 Train Model"):
                with st.spinner("Training..."):
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=int(k_fold))
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

                if problem_type == "Regression":
                    fig_pred = px.scatter(x=y_test, y=test_preds,
                                          labels={"x": "Actual", "y": "Predicted"},
                                          title="Actual vs Predicted", template="plotly_dark")
                    fig_pred.add_shape(type="line",
                                       x0=float(y_test.min()), y0=float(y_test.min()),
                                       x1=float(y_test.max()), y1=float(y_test.max()),
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

        except Exception as e:
            st.error(f"Training error: {e}")
    else:
        st.info("👈 Complete Feature Selection (Tab 4) first.")
