import streamlit as st
import pandas as pd
import os
from pathlib import Path
from rpy2.robjects import r, pandas2ri, conversion, default_converter
import matplotlib.pyplot as plt
import numpy as np

if 'r_env_setup_minimal' not in st.session_state:
    temp_dir = Path.home() / "streamlit_r_temp"
    temp_dir.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(temp_dir)
    try:
        r('library(admixtools)')
        r('library(tidyverse)')
        st.session_state.r_env_setup_minimal = True
    except Exception as e:
        st.error(f"Failed to load R libraries on startup: {e}")
        st.session_state.r_env_setup_minimal = False

def load_predefined_lists(filename):
    lists = {}
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    name, pops_str = line.split(':', 1)
                    pops = [p.strip() for p in pops_str.strip().split(',')]
                    lists[name.strip()] = pops
    return lists

def load_config(filename):
    config = {}
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    key, value = line.split(':', 1)
                    config[key.strip()] = value.strip()
    return config

st.set_page_config(layout="wide", page_title="qpAdm Runner")
st.title("Interactive qpAdm Analysis Interface")

config = load_config('config.txt')
prefix_path = config.get('prefix')
predefined_left = load_predefined_lists('left_list.txt')
predefined_right = load_predefined_lists('right_list.txt')

st.sidebar.title("Configuration")
if prefix_path:
    st.sidebar.info(f"**Dataset Prefix:**\n`{prefix_path}`")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Index File Inspector")

    fam_path = f"{prefix_path}.fam"
    ind_path = f"{prefix_path}.ind"
    actual_file_path = None

    if os.path.exists(fam_path):
        actual_file_path = fam_path
    elif os.path.exists(ind_path):
        actual_file_path = ind_path

    if actual_file_path:
        try:
            with open(actual_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            st.sidebar.download_button(
                label="Download Index File (.fam/.ind)",
                data=file_content.encode('utf-8'),
                file_name=os.path.basename(actual_file_path),
                mime="text/plain",
                help="Click to download the index file to view all population names."
            )
        except Exception as e:
            st.sidebar.error(f"Error reading index file ({os.path.basename(actual_file_path)}): {e}")
    else:
        st.sidebar.warning(f"No index file (.fam or .ind) found.")
else:
    st.sidebar.error("`config.txt` with a `prefix:` entry not found.")

st.header("1. Input Parameters")
target_name = st.text_input("Target Population:", help="Enter the name of your target population.")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Sources (Left)")
    left_options = ["Enter manually"]
    if predefined_left:
        left_options.insert(0, "Use predefined list")
    left_choice = st.radio("Option for Left list:", left_options, horizontal=True, key="left_radio")
    left_pops = []
    if left_choice == "Use predefined list":
        selected_list_name = st.selectbox("Choose Left list:", options=list(predefined_left.keys()))
        if selected_list_name:
            left_pops = predefined_left[selected_list_name]
    else:
        left_input = st.text_area("Enter 'left' populations (one per line):", height=150)
        if left_input:
            left_pops = [pop.strip() for pop in left_input.split('\n') if pop.strip()]
with col2:
    st.subheader("Outgroups (Right)")
    right_options = ["Enter manually"]
    if predefined_right:
        right_options.insert(0, "Use predefined list")
    right_choice = st.radio("Option for Right list:", right_options, horizontal=True, key="right_radio")
    right_pops = []
    if right_choice == "Use predefined list":
        selected_list_name = st.selectbox("Choose Right list:", options=list(predefined_right.keys()))
        if selected_list_name:
            right_pops = predefined_right[selected_list_name]
    else:
        right_input = st.text_area("Enter 'right' outgroups (one per line):", height=150)
        if right_input:
            right_pops = [pop.strip() for pop in right_input.split('\n') if pop.strip()]

st.header("2. Run Analysis")
if st.button("Start qpAdm Analysis"):
    if not st.session_state.get('r_env_setup_minimal'):
        st.error("R environment could not be initialized. Please restart the application.")
    elif not prefix_path:
        st.error("Prefix path not defined in `config.txt`.")
    elif not target_name or not left_pops or not right_pops:
        st.error("ERROR: Please define the Target, Left list, and Right list.")
    else:
        st.info(f"Target: {target_name}")
        st.info(f"Sources (Left): {left_pops}")
        st.info(f"Outgroups (Right): {right_pops}")

        try:
            with conversion.localconverter(default_converter + pandas2ri.converter):
                with st.spinner("Running analysis in R..."):
                    r.assign('prefix', prefix_path)
                    r.assign('target', target_name)
                    r_vector_left = ", ".join([f'"{pop.replace('"', '\\"')}"' for pop in left_pops])
                    r(f'left = c({r_vector_left})')
                    r_vector_right = ", ".join([f'"{pop.replace('"', '\\"')}"' for pop in right_pops])
                    r(f'right = c({r_vector_right})')
                    r('results=qpadm(data=prefix, left=left, right=right, target=target, allsnps=TRUE)')

                    result = r['results']
                    weights = result['weights']
                    rankdrop = result['rankdrop']

            if weights is None:
                st.error("Analysis failed. No feasible model was found.")
            else:
                st.success("Analysis complete!")
                st.header("3. Analysis Result")

                pValue = rankdrop['p'].iloc[0]
                x2 = rankdrop['chisq'].iloc[0]
                weights['zscore'] = weights.apply(lambda row: row['weight'] / row['se'] if row['se'] != 0 else 0, axis=1)

                fig, ax = plt.subplots(figsize=(12, 10))

                plot_data = weights[weights['weight'] > 0]
                wedges, texts, autotexts = ax.pie(plot_data['weight'],
                                                  labels=plot_data['left'],
                                                  autopct='%1.1f%%',
                                                  startangle=90,
                                                  pctdistance=0.85)
                plt.setp(autotexts, size=10, weight="bold", color="white")
                ax.axis('equal')

                stats_text = ""
                stats_text += f"P-Value: {pValue:.4f}\n"
                stats_text += f"χ²: {x2:.4f}\n"
                stats_text += "------------------------------------------------------------\n"
                stats_text += f"{'Source':<40} | {'SE (%)':>7} | {'Z-score':>7}\n"
                stats_text += "------------------------------------------------------------\n"

                for index, row in weights.iterrows():
                    se_percent = row['se'] * 100
                    stats_text += f"{row['left']:<40} | {se_percent:>6.2f}% | {row['zscore']:>7.2f}\n"

                fig.text(0.5, 0.05, stats_text,
                         ha='center', va='bottom',
                         fontsize=12,
                         fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.5", fc='whitesmoke', ec='black', lw=1))

                fig.suptitle(f"Target: {target_name}", fontsize=20, weight="bold")
                plt.subplots_adjust(top=0.9, bottom=0.35)

                st.pyplot(fig)

                @st.cache_data
                def convert_df_to_csv(df_weights, df_rankdrop):
                    df_w = df_weights.pivot_table(index='target', columns='left', values='weight').add_suffix('_weight')
                    df_s = df_weights.pivot_table(index='target', columns='left', values='se').add_suffix('_se')
                    df_z = df_weights.pivot_table(index='target', columns='left', values='zscore').add_suffix('_zscore')
                    df = pd.concat([df_w, df_s, df_z], axis=1).reset_index()
                    df['p-value'] = df_rankdrop['p'].iloc[0]
                    df['x2'] = df_rankdrop['chisq'].iloc[0]
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(weights, rankdrop)
                st.download_button(
                   label="Download full result as CSV",
                   data=csv,
                   file_name=f"results_{target_name}.csv",
                   mime="text/csv",
                )
                st.balloons()

        except Exception as e:
            st.error("A critical error occurred during execution.")
            st.exception(e)
