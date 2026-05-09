import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain_ollama import OllamaLLM
import re
import plotly.express as px
import json

# Set page config for aesthetics
st.set_page_config(page_title="Agentic EDA Pipeline", layout="wide", page_icon="📊")

# --- UI Header ---
st.title("🤖 Agentic AI EDA Pipeline")
st.markdown("Upload a CSV file and get a complete, auto-generated Exploratory Data Analysis.")

# --- Helper Functions ---
def clean_dataframe(df):
    """Attempt to auto-clean malformed DataFrames, e.g., bad headers."""
    if not df.empty:
        first_row = df.iloc[0].astype(str)
        null_count = df.iloc[0].isnull().sum()
        if null_count > len(df.columns) / 2 or "Date" in first_row.values:
            df = df.iloc[1:].reset_index(drop=True)
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except Exception:
                    pass
    
    # Fix pyarrow serialization issues by casting object types properly 
    # to avoid mixed type columns (e.g., int and str in the same column)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    return df


# --- App Logic ---
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Clear session state if a new file is uploaded
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.eda_generated = False
        st.session_state.chat_history = []
        st.session_state.generated_viz_code = None

    try:
        df = pd.read_csv(uploaded_file)
        df = clean_dataframe(df)
        st.success("Data successfully loaded and cleaned!")
        
        with st.expander("Preview Dataset"):
            st.dataframe(df.head())
            
        if "eda_generated" not in st.session_state:
            st.session_state.eda_generated = False
            
        if st.button("Generate EDA"):
            st.session_state.eda_generated = True
            
        if st.session_state.eda_generated:
            st.markdown("---")
            
            # 1. Dataset Overview
            st.header("1. Dataset Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())
            
            # 2. Statistical Parameters
            st.header("2. Statistical Parameters")
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe(include='all'))
            


            # 3. AI Insights
            st.header("3. AI Agent Insights")
            with st.spinner("Agent is analyzing the statistical summary..."):
                try:
                    llm = OllamaLLM(model="mistral")
                    analysis_summary = df.describe(include="all").to_string()
                    insight_prompt = f"""
You are a senior data scientist.

Here are the dataset summary statistics:
{analysis_summary}

Provide:
- Key patterns
- Potential data quality issues
- Interesting correlations (if any are apparent from stats)
- Recommendations for modeling
"""
                    insights = llm.invoke(insight_prompt)
                    st.markdown(insights)
                except Exception as e:
                    st.warning(f"Could not generate AI insights. Error: {e}")

            # 4. Smart Auto-Generated Visualizations (Deterministic - Zero Errors)
            st.header("4. Interactive Visualizations Dashboard")

            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # --- Correlation Heatmap (only if 2+ numeric columns) ---
            if len(num_cols) >= 2:
                st.subheader("📊 Correlation Heatmap")
                corr = df[num_cols].corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title="Numeric Feature Correlations",
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # --- Distributions: Histograms + Boxplots for numeric cols ---
            if num_cols:
                st.subheader("📈 Numeric Column Distributions")
                # Show up to 6 columns to keep dashboard clean
                display_num = num_cols[:6]
                cols_per_row = 2
                for i in range(0, len(display_num), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j, col_name in enumerate(display_num[i:i+cols_per_row]):
                        with row_cols[j]:
                            fig_hist = px.histogram(
                                df, x=col_name,
                                marginal="box",
                                title=f"Distribution of {col_name}",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

            # --- Bar Charts for categorical columns ---
            if cat_cols:
                st.subheader("📊 Categorical Column Breakdown")
                display_cat = cat_cols[:4]
                cols_per_row = 2
                for i in range(0, len(display_cat), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j, col_name in enumerate(display_cat[i:i+cols_per_row]):
                        with row_cols[j]:
                            top_vals = df[col_name].value_counts().head(15).reset_index()
                            top_vals.columns = [col_name, "Count"]
                            fig_bar = px.bar(
                                top_vals, x=col_name, y="Count",
                                title=f"Top Values in {col_name}",
                                template="plotly_white",
                                color="Count",
                                color_continuous_scale="Blues"
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

            if not num_cols and not cat_cols:
                st.info("No suitable columns found for visualization.")

            # 5. Chat with your Data
            st.markdown("---")
            st.header("5. Chat with your Dataset")
            st.markdown("Ask questions about the dataset's summary statistics, structure, or patterns.")
            
            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
                
            # Display chat messages
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
            # Chat input
            if prompt := st.chat_input("Ask a question about your dataset..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                with st.chat_message("assistant"):
                    try:
                        llm = OllamaLLM(model="mistral")
                        
                        # Construct context from dataframe
                        context = f"""
You are a helpful data science assistant. Answer the user's question based strictly on the following dataset summary.
Do not hallucinate answers. If you cannot answer based on this context, state that clearly.

Data Types:
{df.dtypes.to_string()}

Summary Statistics:
{df.describe(include='all').to_string()}

First 5 Rows:
{df.head().to_string()}

Question: {prompt}
"""
                        # Stream the response directly to the UI
                        response = st.write_stream(llm.stream(context))
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                    
    except Exception as e:
        st.error(f"Error loading CSV: {e}")