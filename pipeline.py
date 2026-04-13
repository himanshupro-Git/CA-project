import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
if 'df' not in st.session_state:
	st.session_state.df = None

if 'target_col' not in st.session_state:
	st.session_state.target_col = None

# Tabs
tabs = st.tabs([
	"📥 Data Input", 
	"📊 EDA", 
	"🛠 Engineering", 
	"🎯 Feature Selection", 
	"🤖 Training & Tuning"
])

# =========================
# TAB 1: DATA INPUT + PCA
# =========================
with tabs[0]:
	uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

	if uploaded_file:
		st.session_state.df = pd.read_csv(uploaded_file)
		df = st.session_state.df

		col1, col2 = st.columns([1, 2])
		with col1:
			target_col = st.selectbox("Select Target Feature", df.columns)
			st.session_state.target_col = target_col

			features = st.multiselect(
				"Select Features for PCA/Analysis",
				[c for c in df.columns if c != target_col],
				default=[c for c in df.columns if c != target_col][:5]
			)

		# ✅ FIXED PCA LOGIC
		if features:
			numeric_df = df[features].select_dtypes(include=[np.number]).dropna()

			if numeric_df.shape[1] >= 2:
				scaled_data = StandardScaler().fit_transform(numeric_df)

				pca = PCA(n_components=2)
				components = pca.fit_transform(scaled_data)

				color_data = df.loc[numeric_df.index, target_col].astype(str)

				fig_pca = px.scatter(
					components,
					x=0,
					y=1,
					color=color_data,
					title="Data Shape (PCA 2D Projection)",
					labels={'0': 'PC1', '1': 'PC2'},
					template="plotly_dark"
				)

				st.plotly_chart(fig_pca, use_container_width=True)

			else:
				st.warning("⚠️ Please select at least 2 numeric features for PCA.")

# =========================
# TAB 2: EDA
# =========================
with tabs[1]:
	if st.session_state.df is not None:
		st.header("Exploratory Data Analysis")

		col1, col2 = st.columns(2)

		with col1:
			st.write("**Dataset Summary**")
			st.write(st.session_state.df.describe())

		with col2:
			st.write("**Correlation Matrix**")
			corr = st.session_state.df.corr(numeric_only=True)
			fig_corr = px.imshow(corr, text_auto=True, aspect="auto")
			st.plotly_chart(fig_corr)

# =========================
# TAB 3: ENGINEERING
# =========================
with tabs[2]:
	if st.session_state.df is not None:
		df = st.session_state.df.copy()

		st.subheader("Missing Value Imputation")
		method = st.selectbox("Imputation Method", ["None", "Mean", "Median", "Mode"])

		if method != "None":
			for col in df.select_dtypes(include=[np.number]).columns:
				if method == "Mean":
					df[col].fillna(df[col].mean(), inplace=True)
				elif method == "Median":
					df[col].fillna(df[col].median(), inplace=True)
				elif method == "Mode":
					df[col].fillna(df[col].mode()[0], inplace=True)

			st.success("Imputation Applied")

		st.subheader("Outlier Detection")
		outlier_method = st.selectbox("Method", ["IQR", "Isolation Forest"])

if outlier_method == "IQR":
	num_df = df.select_dtypes(include=[np.number])

	if not num_df.empty:
		Q1 = num_df.quantile(0.25)
		Q3 = num_df.quantile(0.75)
		IQR = Q3 - Q1

		outliers = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)
	else:
		outliers = np.array([False] * len(df))

else:
	num_df = df.select_dtypes(include=[np.number])

	if not num_df.empty:
		iso = IsolationForest(contamination=0.1)
		outliers = iso.fit_predict(num_df) == -1
	else:
		outliers = np.array([False] * len(df))

		st.warning(f"Detected {sum(outliers)} outliers")

		if st.button("Remove Outliers"):
			st.session_state.df = df[~outliers]
			st.rerun()

# =========================
# TAB 4: FEATURE SELECTION
# =========================
with tabs[3]:
	if st.session_state.df is not None and st.session_state.target_col:
		df = st.session_state.df.dropna()
		target_col = st.session_state.target_col

		X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
		y = df[target_col]

		method = st.radio("Selection Criterion", ["Variance Threshold", "Information Gain"])

		if method == "Variance Threshold":
			threshold = st.slider("Threshold", 0.0, 1.0, 0.0)
			sel = VarianceThreshold(threshold=threshold)
			sel.fit(X)
			selected_features = X.columns[sel.get_support()]
		else:
			if problem_type == "Classification":
				scores = mutual_info_classif(X, y)
			else:
				scores = mutual_info_regression(X, y)

			feat_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
			st.bar_chart(feat_scores)
			selected_features = feat_scores.index[:5]

		st.write("Selected Features:", list(selected_features))
		st.session_state.final_features = selected_features

# =========================
# TAB 5: TRAINING
# =========================
with tabs[4]:
	if 'final_features' in st.session_state:
		target_col = st.session_state.target_col

		test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

		X_train, X_test, y_train, y_test = train_test_split(
			st.session_state.df[st.session_state.final_features],
			st.session_state.df[target_col],
			test_size=test_size
		)

		model_choice = st.selectbox("Model", ["Linear/Logistic Regression", "SVM", "Random Forest"])
		k_fold = st.number_input("K-Fold", 2, 10, 5)

		if model_choice == "Random Forest":
			model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
			params = {"n_estimators": [10, 50, 100]}
		elif model_choice == "SVM":
			model = SVC() if problem_type == "Classification" else SVR()
			params = {"C": [0.1, 1, 10]}
		else:
			model = LogisticRegression() if problem_type == "Classification" else LinearRegression()
			params = {}

		if st.button("Train"):
			scaler = StandardScaler()
			X_train_scaled = scaler.fit_transform(X_train)
			X_test_scaled = scaler.transform(X_test)

			cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=k_fold)
			model.fit(X_train_scaled, y_train)

			test_preds = model.predict(X_test_scaled)

			st.write(f"CV Score: {cv_scores.mean():.4f}")

			if problem_type == "Classification":
				st.write(f"Accuracy: {accuracy_score(y_test, test_preds):.4f}")
				st.text(classification_report(y_test, test_preds))
			else:
				st.write(f"R2: {r2_score(y_test, test_preds):.4f}")
				st.write(f"MSE: {mean_squared_error(y_test, test_preds):.4f}")

		if st.checkbox("Grid Search") and params:
			grid = GridSearchCV(model, params, cv=3)
			grid.fit(X_train, y_train)

			st.write("Best Params:", grid.best_params_)
			st.write("Best Score:", grid.best_score_)
