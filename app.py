import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from nltk import bigrams, trigrams
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Set page config
st.set_page_config(layout="wide")

# Title and description
st.title("Fake News Analysis Dashboard")
st.markdown("Analyze fake news patterns with interactive visualizations and filters.")

# Load data
df = pd.read_csv('IFND.csv', encoding='latin-1')

# Rename columns to match requirements
df = df.rename(columns={
    'Statement': 'title',
    'Category': 'subject',
    'Date': 'date',
    'Label': 'label',
    'Web': 'place'
})

# Add text column if not present (using title as text)
df['text'] = df['title']

# Map label to Real/Fake (handling both 'TRUE'/'FALSE' and 'Real'/'Fake' formats)
df['label'] = df['label'].astype(str).str.strip().apply(
    lambda x: 'Real' if x.upper() == 'TRUE' else ('Fake' if x.upper() in ['FALSE', 'FAKE'] else x)
)

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'], format='%b-%y', errors='coerce')

# Add word count for analysis
df['word_count'] = df['text'].str.split().str.len()

# Sidebar filters
st.sidebar.header("Filters")

subjects = df['subject'].unique()
selected_subject = st.sidebar.selectbox("Select Subject", ['All'] + list(subjects))

places = df['place'].unique()
selected_place = st.sidebar.selectbox("Select Place", ['All'] + list(places))

label_filter = st.sidebar.radio("Filter by Label", ['Both', 'Real', 'Fake'])

# Apply filters
filtered_df = df.copy()
if selected_subject != 'All':
    filtered_df = filtered_df[filtered_df['subject'] == selected_subject]
if selected_place != 'All':
    filtered_df = filtered_df[filtered_df['place'] == selected_place]
if label_filter != 'Both':
    filtered_df = filtered_df[filtered_df['label'] == label_filter]

# KPI Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Articles", len(filtered_df))
with col2:
    fake_pct = (filtered_df['label'] == 'Fake').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    st.metric("% Fake News", f"{fake_pct:.1f}%")
with col3:
    avg_word_count = filtered_df['text'].str.split().str.len().mean() if len(filtered_df) > 0 else 0
    st.metric("Avg Word Count", f"{avg_word_count:.0f}")

# Visualizations
st.header("Visualizations")

# Chart filter for Real/Fake
chart_label_filter = st.radio("Filter Charts by Label", ['All', 'Real', 'Fake'], horizontal=True, key="chart_filter")

# Use filtered data for charts based on selection
if chart_label_filter == 'All':
    chart_df = df.copy()
else:
    chart_df = df[df['label'] == chart_label_filter].copy()

# ===== MAIN DASHBOARD LAYOUT =====
# Top Row: KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Articles", len(filtered_df))
with col2:
    fake_pct = (filtered_df['label'] == 'Fake').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    st.metric("% Fake News", f"{fake_pct:.1f}%")
with col3:
    avg_word_count = filtered_df['text'].str.split().str.len().mean() if len(filtered_df) > 0 else 0
    st.metric("Avg Word Count", f"{avg_word_count:.0f}")

# Middle Row: Pie Chart and Subject Bar Chart
col4, col5 = st.columns(2)

with col4:
    # Pie Chart: Real vs Fake
    label_counts = chart_df['label'].value_counts().reset_index(name='count').rename(columns={'index': 'label'})
    fig_pie = px.pie(label_counts, names='label', values='count', title=f"Real vs Fake Distribution ({chart_label_filter} Data)")
    st.plotly_chart(fig_pie)

with col5:
    # Bar Chart: Count by Subject
    subject_counts = chart_df['subject'].value_counts().reset_index(name='count').rename(columns={'index': 'subject'})
    fig_bar = px.bar(subject_counts, x='subject', y='count', title=f"News by Subject ({chart_label_filter} Data)")
    st.plotly_chart(fig_bar)

# Bottom Row: Word Cloud and Time Series
col6, col7 = st.columns(2)

with col6:
    # Word Cloud
    st.subheader("Word Cloud")
    all_text = ' '.join(chart_df['text'].dropna())
    if all_text:
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_text)
        fig_wc, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc)
    else:
        st.info("No text data available")

with col7:
    # Time-Series Line Chart
    if chart_df['date'].notna().any():
        time_series = chart_df.groupby(chart_df['date'].dt.to_period('M')).size().reset_index(name='count')
        time_series['date'] = time_series['date'].dt.to_timestamp()
        fig_line = px.line(time_series, x='date', y='count', title=f"News Volume Over Time ({chart_label_filter} Data)")
        st.plotly_chart(fig_line)
    else:
        st.info("No valid dates available")

# ===== ADDITIONAL VISUALIZATIONS =====
st.header("📊 Advanced Analytics")

# Heatmap: Subject vs Label
subject_label_crosstab = pd.crosstab(chart_df['subject'], chart_df['label'])
fig_heatmap = go.Figure(data=go.Heatmap(
    z=subject_label_crosstab.values,
    x=subject_label_crosstab.columns,
    y=subject_label_crosstab.index,
    colorscale='YlOrRd',
    text=subject_label_crosstab.values,
    texttemplate='%{text}',
    textfont={"size": 10}
))
fig_heatmap.update_layout(
    title=f"Subject vs Label Distribution Heatmap ({chart_label_filter} Data)",
    xaxis_title="Label",
    yaxis_title="Subject",
    height=500
)
st.plotly_chart(fig_heatmap)

# Scatter Plot: Date vs Subject by Label
if chart_df['date'].notna().any():
    # Group by date and subject to get counts
    scatter_data = chart_df.groupby([chart_df['date'].dt.to_period('M'), 'subject', 'label']).size().reset_index(name='count')
    scatter_data['date'] = scatter_data['date'].dt.to_timestamp()
    
    fig_scatter = px.scatter(scatter_data, 
                           x='date', 
                           y='subject', 
                           color='label',
                           size='count',
                           title=f"Article Distribution Over Time by Subject ({chart_label_filter} Data)",
                           labels={'date': 'Date', 'subject': 'Subject', 'label': 'Label', 'count': 'Article Count'})
    fig_scatter.update_layout(height=600)
    st.plotly_chart(fig_scatter)
else:
    st.info("No valid dates available to plot scatter plot.")

# Scatter Plot: Real vs Fake News Timeline
if chart_df['date'].notna().any():
    timeline_data = chart_df[['date', 'label']].copy()
    # Sort by date
    timeline_data = timeline_data.sort_values('date')
    
    fig_timeline = px.scatter(timeline_data, 
                            x='date', 
                            y='label', 
                            color='label',
                            color_discrete_map={'Real': 'green', 'Fake': 'red'},
                            title=f"Real vs Fake News Timeline ({chart_label_filter} Data)",
                            labels={'date': 'Date', 'label': 'News Type'})
    fig_timeline.update_layout(height=400, yaxis_title="News Type")
    st.plotly_chart(fig_timeline)
else:
    st.info("No valid dates available to plot timeline.")

# Horizontal Bar Chart: Top 10 Places
top_places = chart_df['place'].value_counts().reset_index(name='count').rename(columns={'index': 'place'}).head(10)
fig_hbar = px.bar(top_places, x='count', y='place', orientation='h', title=f"Top 10 Places Mentioned ({chart_label_filter} Data)")
st.plotly_chart(fig_hbar)

# ===== STATISTICAL & DISTRIBUTION PLOTS =====
st.header("📊 Statistical & Distribution Analysis")

# Article Length Distribution
st.subheader("Article Length Distribution")
chart_df['word_count'] = chart_df['text'].str.split().str.len()

fig_hist = px.histogram(chart_df, x='word_count', color='label', 
                       marginal='box', nbins=50,
                       title=f"Article Word Count Distribution ({chart_label_filter} Data)",
                       labels={'word_count': 'Word Count', 'label': 'News Type'})
st.plotly_chart(fig_hist)

# Subject/Category Breakdown (Stacked Bar)
st.subheader("Subject Breakdown by News Type")
subject_label_counts = pd.crosstab(chart_df['subject'], chart_df['label'])
fig_stacked = go.Figure()
for label in subject_label_counts.columns:
    fig_stacked.add_trace(go.Bar(
        name=label,
        x=subject_label_counts.index,
        y=subject_label_counts[label],
    ))
fig_stacked.update_layout(
    barmode='stack',
    title=f"Subject Distribution by News Type ({chart_label_filter} Data)",
    xaxis_title="Subject",
    yaxis_title="Count"
)
st.plotly_chart(fig_stacked)

# ===== LINGUISTIC & TEXTUAL ANALYSIS =====
st.header("📝 Linguistic & Textual Analysis")

# Comparative Word Clouds
st.subheader("Word Clouds: Real vs Fake News")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Real News")
    real_text = ' '.join(chart_df[chart_df['label'] == 'Real']['text'].dropna())
    if real_text:
        wordcloud_real = WordCloud(width=400, height=300, background_color='white').generate(real_text)
        fig_wc_real, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(wordcloud_real, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc_real)
    else:
        st.info("No real news data available")

with col2:
    st.subheader("Fake News")
    fake_text = ' '.join(chart_df[chart_df['label'] == 'Fake']['text'].dropna())
    if fake_text:
        wordcloud_fake = WordCloud(width=400, height=300, background_color='white').generate(fake_text)
        fig_wc_fake, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(wordcloud_fake, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc_fake)
    else:
        st.info("No fake news data available")

# Top N-Grams
st.subheader("Top Bigrams in Articles")

# Function to get bigrams
def get_top_ngrams(text_series, n=20):
    all_text = ' '.join(text_series.dropna()).lower()
    tokens = nltk.word_tokenize(all_text)
    bigram_list = list(bigrams(tokens))
    bigram_counts = Counter(bigram_list)
    return bigram_counts.most_common(n)

bigram_data = get_top_ngrams(chart_df['text'])
if bigram_data:
    bigrams_df = pd.DataFrame(bigram_data, columns=['bigram', 'count'])
    bigrams_df['bigram'] = bigrams_df['bigram'].apply(lambda x: ' '.join(x))
    
    fig_bigrams = px.bar(bigrams_df.head(15), x='count', y='bigram', orientation='h',
                        title=f"Top 15 Bigrams ({chart_label_filter} Data)")
    st.plotly_chart(fig_bigrams)

# Sentiment Analysis
st.subheader("Sentiment Polarity Distribution")

# Calculate sentiment
chart_df['sentiment'] = chart_df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

fig_sentiment = px.box(chart_df, x='label', y='sentiment', 
                      title=f"Sentiment Polarity by News Type ({chart_label_filter} Data)",
                      labels={'label': 'News Type', 'sentiment': 'Sentiment Polarity'})
st.plotly_chart(fig_sentiment)

# ===== MACHINE LEARNING RESULTS =====
st.header("🤖 Machine Learning Model Results")

# Note: These are placeholder visualizations. In a real scenario, you'd have trained model predictions.
st.info("Note: The following ML visualizations use simulated data for demonstration. Replace with actual model results.")

# Confusion Matrix (simulated)
st.subheader("Confusion Matrix")
# Simulate predictions (for demo)
np.random.seed(42)
y_true = chart_df['label'].map({'Real': 0, 'Fake': 1})
y_pred = y_true.copy()
# Add some noise to simulate errors
noise_indices = np.random.choice(len(y_pred), size=int(len(y_pred)*0.1), replace=False)
y_pred.iloc[noise_indices] = 1 - y_pred.iloc[noise_indices]

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Real', 'Actual Fake'], columns=['Predicted Real', 'Predicted Fake'])

fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto",
                  title="Confusion Matrix (Simulated)",
                  labels=dict(x="Predicted", y="Actual"))
st.plotly_chart(fig_cm)

# Feature Importance (simulated)
st.subheader("Feature Importance")
features = ['word_count', 'sentiment', 'subject_polarity', 'title_length']
importance_scores = np.random.rand(len(features))
importance_df = pd.DataFrame({'feature': features, 'importance': importance_scores})
importance_df = importance_df.sort_values('importance', ascending=True)

fig_importance = px.bar(importance_df, x='importance', y='feature', orientation='h',
                       title="Feature Importance (Simulated)")
st.plotly_chart(fig_importance)

# ROC-AUC Curve (simulated)
st.subheader("ROC-AUC Curve")
# Simulate probabilities
y_proba = np.random.rand(len(y_true))
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
fig_roc.update_layout(title='ROC-AUC Curve (Simulated)', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
st.plotly_chart(fig_roc)

# Data Table
with st.expander("View Raw Data"):
    st.dataframe(filtered_df)