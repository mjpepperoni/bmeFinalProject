import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


st.set_page_config(
    page_title = "Parkinson's Subtype Discovery Explorer",
    layout = "wide"
)


@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"

    df = pd.read_csv(url)
    df = df.dropna()

    return df


df = load_data()

all_features = [
    "age",
    "motor_UPDRS",
    "total_UPDRS",
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Jitter:DDP",
    "Shimmer",
    "Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:APQ11",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "PPE"
]

default_features = [
    "age",
    "motor_UPDRS",
    "total_UPDRS",
    "Jitter(%)",
    "Shimmer",
    "NHR",
    "HNR"
]


st.title("Parkinson's Subtype Discovery Explorer")

st.write(
    "This app uses unsupervised learning to explore whether Parkinson's patients form hidden groups "
    "based on disease severity and voice measurement features. Instead of predicting a label, the app looks "
    "for patterns in the data."
)

st.info(
    "How to use the app: choose the features you want to cluster on, choose the number of clusters, "
    "and then compare the plots and tables to see what makes each patient group different."
)


st.sidebar.header("App Controls")

selected_features = st.sidebar.multiselect(
    "Select features for clustering",
    all_features,
    default = default_features
)
st.sidebar.write(
    "Guidance:\n"
    "- Choose 3–7 features for best results\n"
    "- Include UPDRS scores to capture disease severity\n"
    "- Add voice features (jitter, shimmer, NHR, HNR) to capture speech patterns\n"
    "- Different feature selections will produce different clusters\n"
)

st.sidebar.write(
    "These features are used to decide which patients are similar to each other. "
    "Changing the features can change the clusters."
)

cluster_count = st.sidebar.slider(
    "Number of clusters",
    min_value = 2,
    max_value = 6,
    value = 3
)

st.sidebar.write(
    "K-Means is an unsupervised method that groups similar patient recordings together. "
    "Each cluster is a group of recordings with similar feature patterns. "
    "A smaller number of clusters creates broader, more general groups, while a larger number of clusters creates more specific, detailed groups."
)
st.sidebar.write(
    "For example, with fewer clusters, patients may be grouped into 'mild' vs 'severe'. "
    "With more clusters, the app may separate patients into more specific subtypes like 'mild', 'moderate', and 'severe'."
)

if len(selected_features) < 2:
    st.warning("Please select at least two features.")
    st.stop()


st.subheader("Dataset Overview")

st.write(
    "Each row in the dataset is a Parkinson's telemonitoring voice recording. "
    "The dataset includes clinical severity scores such as UPDRS and voice measurements such as jitter, shimmer, NHR, and HNR."
)

st.write(
    "These features describe both disease severity and voice characteristics, "
    "which are commonly affected in Parkinson’s disease."
)

st.write(
    "Feature definitions:\n\n"
    "- age: Age of the patient\n"
    "- motor_UPDRS: Clinical score measuring motor symptom severity (higher = worse movement symptoms)\n"
    "- total_UPDRS: Overall Parkinson’s severity score (higher = more severe disease)\n\n"
    "Voice features (Jitter and Shimmer have different variants in the dataset):\n"
    "- Jitter measures variation in voice pitch (frequency). Higher values indicate a more unstable voice.\n"
    "- Shimmer measures variation in voice loudness (amplitude). Higher values indicate more instability in volume.\n"
    "- NHR (Noise-to-Harmonics Ratio): Higher values indicate more noise in the voice signal.\n"
    "- HNR (Harmonics-to-Noise Ratio): Higher values indicate a clearer, more stable voice.\n\n"
    "Advanced voice features:\n"
    "- RPDE measures how unpredictable the voice signal is over time.\n"
    "- DFA measures long-term patterns and complexity in the voice signal.\n"
    "- PPE measures irregularity in voice pitch.\n"
)

st.write(
    "UPDRS stands for Unified Parkinson’s Disease Rating Scale. "
    "It is a clinical score used to measure the severity of Parkinson’s disease symptoms. "
    "Higher values indicate more severe symptoms."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Patients", df["subject#"].nunique())

with col3:
    st.metric("Selected features", len(selected_features))


st.write("Preview of the selected data")

st.dataframe(df[["subject#"] + selected_features].head())


X = df[selected_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters = cluster_count, random_state = 1, n_init = 10)
clusters = kmeans.fit_predict(X_scaled)

plot_df = df.copy()
plot_df["cluster"] = clusters.astype(str)
plot_df["PC1"] = X_pca[:, 0]
plot_df["PC2"] = X_pca[:, 1]


st.subheader("PCA Visualization of Patient Recordings")

st.write(
    "PCA (Principal Component Analysis) combines the selected features into two summary variables called PC1 and PC2. "
    "PC1 captures the largest variation in the data, and PC2 captures the next largest variation. "
    "These components allow the data to be visualized in two dimensions. "
    "Each point is one recording, and points that are close together have similar feature patterns."
)

pca_fig = px.scatter(
    plot_df,
    x = "PC1",
    y = "PC2",
    color = "cluster",
    hover_data = ["subject#", "age", "motor_UPDRS", "total_UPDRS"],
    title = "PCA Plot Colored by K-Means Cluster"
)

st.plotly_chart(pca_fig, use_container_width = True)

st.write(
    "PC1 explains " + str(round(pca.explained_variance_ratio_[0] * 100, 2)) +
    "% of the variation in the selected features. PC2 explains " +
    str(round(pca.explained_variance_ratio_[1] * 100, 2)) + "%."
)


st.subheader("Cluster Summary Table")

st.write(
    "This table shows the average value of each selected feature within each cluster. "
    "It helps describe what each possible subtype looks like."
)

summary_df = plot_df.groupby("cluster")[selected_features].mean().reset_index()
st.dataframe(summary_df)


st.subheader("Cluster Comparison Chart")

st.write(
    "This chart compares the cluster averages for one selected feature. "
    "Use it to see whether a feature is higher or lower in different clusters."
)

feature_to_compare = st.selectbox(
    "Select feature to compare across clusters",
    selected_features
)

comparison_fig = px.bar(
    summary_df,
    x = "cluster",
    y = feature_to_compare,
    color = "cluster",
    title = "Average " + feature_to_compare + " by Cluster"
)

st.plotly_chart(comparison_fig, use_container_width = True)


st.subheader("What Features Define the Clusters?")

st.write(
    "In clustering there's no true model feature importance like there is in supervised prediction. "
    "Here, feature importance is estimated by how much the cluster averages differ from each other. "
    "Features with larger differences across clusters are more useful for describing the discovered groups."
)

importance_rows = []

for feature in selected_features:
    feature_means = summary_df[feature]
    importance = feature_means.max() - feature_means.min()

    importance_rows.append({
        "feature" : feature,
        "cluster_mean_range" : importance
    })

importance_df = pd.DataFrame(importance_rows)
importance_df = importance_df.sort_values("cluster_mean_range", ascending = False)

importance_fig = px.bar(
    importance_df,
    x = "cluster_mean_range",
    y = "feature",
    orientation = "h",
    title = "Features That Differ Most Across Clusters"
)

st.plotly_chart(importance_fig, use_container_width = True)

st.dataframe(importance_df)


st.subheader("Cluster Size")

st.write(
    "This chart shows how many recordings are assigned to each cluster. "
    "Very small clusters may represent unusual recordings or a narrow subtype."
)

size_df = plot_df["cluster"].value_counts().reset_index()
size_df.columns = ["cluster", "count"]

size_fig = px.bar(
    size_df,
    x = "cluster",
    y = "count",
    color = "cluster",
    title = "Number of Recordings in Each Cluster"
)

st.plotly_chart(size_fig, use_container_width = True)


st.subheader("Interpretation")

st.write(
    "The goal of this app is to explore possible Parkinson's subtypes. "
    "If one cluster has higher UPDRS scores, it may represent a more severe symptom group. "
    "If another cluster has lower UPDRS scores or different voice measurements, it may represent a milder or different voice-pattern group."
)

st.write(
    "Because this is unsupervised learning, the clusters aren't guaranteed clinical categories. "
    "They are exploratory groups based on similarity in the selected features. "
    "The app should be used to understand patterns in the dataset, not to diagnose or classify individual patients."
)
