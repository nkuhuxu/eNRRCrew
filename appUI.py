import autogen
import streamlit as st
from typing_extensions import Annotated
from typing import Dict, Optional, Union, Callable
from autogen import AssistantAgent, UserProxyAgent, Agent
from dotenv import load_dotenv, set_key
import os, time


class StreamlitAssistantAgent(AssistantAgent):
    """
    Wrapper for AutoGen's AssistantAgent using Streamlit for UI
    """
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        # Display the message being sent in Streamlit
        st.write(f'*{self.name} is sending a message to "{recipient.name}":*')
        st.write(message)
        # Call the parent class's send method
        return super(StreamlitAssistantAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )


class StreamlitUserProxyAgent(UserProxyAgent):
    """
    Wrapper for AutoGen's UserProxyAgent using Streamlit to simplify the UI by adding interactions
    """
    def get_human_input(self, prompt: str) -> str:
        # Use a counter to generate unique keys
        if 'input_counter' not in st.session_state:
            st.session_state.input_counter = 0
        st.session_state.input_counter += 1
        key_suffix = f"_{st.session_state.input_counter}"
        
        if prompt.startswith(
            "Provide feedback to chat_manager. Press enter to skip and use auto-reply"
        ):
            st.write("Continue or exit the conversation?")
            
            action = st.radio(
                "Select an action:",
                ["✅ Continue", "🔚 Exit Conversation"],
                key=f"action_select{key_suffix}"
            )
            with st.spinner('Pausing for a moment...'):
                time.sleep(5)
            if action == "✅ Continue":
                # Proceed with continuation
                return ""
            elif action == "🔚 Exit Conversation":
                # Exit the conversation
                return "exit"
        else:
            # Get user input with a unique key
            reply = st.text_input(prompt, key=f"input{key_suffix}")
            return reply.strip()


    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # Display the message being sent in Streamlit
        # st.write(f'*{self.name} is sending a message to "{recipient.name}":*')
        # st.write(message)
        # Call the parent class's send method
        super(StreamlitUserProxyAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )


# Import other necessary modules
from graphrag.query.cli import run_global_search, run_local_search
from autogen.coding import LocalCommandLineCodeExecutor
import os
import pandas as pd
import numpy as np
import random
from joblib import load
from sklearn.preprocessing import StandardScaler
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid


def preprocess_new_data(new_data):
    # 设置随机种子以保持一致性
    RANDOM_SEED = 49
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    # 1. 创建元素组合描述符
    columns_to_combine = [
        'Elements of electrocatalyst_0', 'Elements of electrocatalyst_1', 
        'Elements of electrocatalyst_2', 'Elements of electrocatalyst_3', 
        'Elements of electrocatalyst_4', 'Elements of electrocatalyst_5', 
        'Elements of electrocatalyst_6'
    ]
    # 创建化学式
    new_data['formula'] = new_data[columns_to_combine].apply(
        lambda row: ''.join(row.dropna().astype(str)).replace(' ', ''), 
        axis=1
    )
    # 使用matminer进行特征化
    new_data = StrToComposition().featurize_dataframe(new_data, "formula")
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    new_data = ep_feat.featurize_dataframe(new_data, col_id="composition")
    # 2. 清理数据
    def clean_data(df):
        df = df.drop(columns=[
            'Faraday efficiency', 'Applied Potential (Faraday Efficiency)', 
            'Elements of electrocatalyst_0', 'Elements of electrocatalyst_1', 
            'Elements of electrocatalyst_2', 'Elements of electrocatalyst_3', 
            'Elements of electrocatalyst_4', 'Elements of electrocatalyst_5', 
            'Elements of electrocatalyst_6', 'Yield_mg', 'Yield_cm',
            'Element_0_ionization_energies', 'Element_1_ionization_energies', 
            'Element_2_ionization_energies', 'Element_3_ionization_energies', 
            'Element_0_electron_configuration', 'Element_1_electron_configuration', 
            'Element_2_electron_configuration', 'Element_3_electron_configuration', 
            'Element_4_ionization_energies', 'Element_4_electron_configuration', 
            'Element_5_ionization_energies', 'Element_5_electron_configuration', 
            'Element_6_ionization_energies', 'Element_6_electron_configuration',
            'pH_nan', 'Electrolyte without concentration_nan', 'N-15 labeling_nan'
        ], errors='ignore')
        return df
    new_data = clean_data(new_data)
    # 删除特定的判断列和其他不需要的列
    new_data = new_data.drop(columns=[
        'Judge_MOF', 'Judge_COF', 'Judge_Hydroxide_Oxyhydroxide', 'Judge_Spinel', 
        'Judge_Perovskite', 'Judge_Prussian_Blue', 'Judge_Ultrasound', 'Judge_c3n4', 
        'Judge_Steel', 'Judge_Carbon_Cloth', 'Judge_Hydrothermal', 'Judge_Spray', 
        'Judge_Phys_Chem_Deposition', 'Judge_Electrochemical_Deposition', 'pH_ionic liquid', 
        'pH_khco3', 'pH_li tfsi', 'pH_nabf4', 'Electrolyte without concentration_kclo4', 
        'Electrolyte without concentration_licl', 'Electrolyte without concentration_lioh', 
        'Electrolyte without concentration_naoh'
    ], errors='ignore')
    # 删除原子特征列
    columns_to_drop = [
        col for col in new_data.columns if any(x in col for x in 
        ['atomic_number', 'period', 'group_number', 'atomic_mass', 
         'ionization_potential', 'electronegativity', 'atomic_radius'])
    ]
    new_data = new_data.drop(columns=columns_to_drop, errors='ignore')
    # 删除电催化剂列
    new_data = new_data.drop(columns=['Electrocatalyst'], errors='ignore')
    # 3. 处理形态学特征
    def categorize_structure(morphology):
        """
        根据形貌数据对其所属的结构类型进行分类
        """
        if pd.isna(morphology):  # 检查缺失值
            return 'No values'
        
        morphology = str(morphology).lower()  # 转换为小写
        if 'nanoparticle' in morphology or 'quantum dots' in morphology or 'nanodots' in morphology: 
            return 'Nanoparticles'
        elif 'nanorod' in morphology:
            return 'Nanorods'
        elif 'nanowire' in morphology:
            return 'Nanowires'
        elif 'nanosheet' in morphology or '2d' in morphology:
            return 'Nanosheets'
        elif 'nanofiber' in morphology:
            return 'Nanofibers'
        elif 'nanocube' in morphology:
            return 'Nanocubes'
        elif 'hollow' in morphology:
            return 'Hollow Structures'
        elif 'porous' in morphology:
            return 'Porous Structures'
        elif 'hybrid' in morphology or 'composite' in morphology:
            return 'Hybrids and Composites'
        elif 'nanotube' in morphology:
            return 'Nanotubes'
        elif 'nanowall' in morphology:
            return 'Nanowalls'
        else:
            return 'Other'

    structure_types = [
        'Hollow Structures', 'Hybrids and Composites', 'Nanocubes', 'Nanofibers', 'Nanoparticles', 
        'Nanorods', 'Nanosheets', 'Nanotubes', 'Nanowalls', 
        'Nanowires', 'Other', 'Porous Structures'
    ]

    # 初始化所有结构类型的列为 0
    for structure in structure_types:
        new_data[f'Structure Type_{structure}'] = 0

    # 遍历数据并更新对应列为 1
    for index, row in new_data.iterrows():
        structure_type = categorize_structure(row['Morphology of electrocatalyst'])
        if f'Structure Type_{structure_type}' in new_data.columns:
            new_data.at[index, f'Structure Type_{structure_type}'] = 1
    # 删除形态学列
    new_data = new_data.drop(columns=['Morphology of electrocatalyst'], errors='ignore')
    # 删除不需要的列
    new_data = new_data.drop(columns=['formula', 'composition'], errors='ignore')
    # 转换数据类型
    #new_data = new_data.astype('float64')
    # 删除以Judge开头的列
    new_data = new_data.loc[:, ~new_data.columns.str.startswith('Judge')]
    return new_data

def predict_fe_high_low(new_data_path, model_paths):
    """
    Predict whether new data points have high or low Faraday Efficiency (FE).
    
    Parameters:
    - new_data_path (str): Path to the new data CSV file.
    - model_paths (dict): Dictionary containing paths to saved models and featurizers.
        Example:
            {
                'scaler': './models/scaler.joblib',
                'nearest_centroid': './models/nearest_centroid.joblib',
                'str_to_comp': './models/str_to_comp.joblib',
                'element_property': './models/element_property.joblib',
                'features': './models/features.joblib'
            }
    
    Returns:
    - predictions_df (pd.DataFrame): DataFrame with original data and FE prediction.
    """
    # Load saved models and featurizers
    scaler = load(model_paths['scaler'])
    nearest_centroid = load(model_paths['nearest_centroid'])
    str_to_comp = load(model_paths['str_to_comp'])
    ep_feat = load(model_paths['element_property'])
    features = load(model_paths['features'])
    
    # Load new data
    new_data = pd.read_csv(new_data_path)
    
    # Combine elemental columns to form 'formula'
    columns_to_combine = [
        'Elements of electrocatalyst_0', 'Elements of electrocatalyst_1', 
        'Elements of electrocatalyst_2', 'Elements of electrocatalyst_3', 
        'Elements of electrocatalyst_4', 'Elements of electrocatalyst_5', 
        'Elements of electrocatalyst_6'
    ]
    
    new_data['formula'] = new_data[columns_to_combine].apply(
        lambda row: ''.join(row.dropna().astype(str)).replace(' ', ''), axis=1
    )
    
    # Featurize the formula
    new_data = str_to_comp.featurize_dataframe(new_data, "formula")
    
    # Featurize composition
    new_data = ep_feat.featurize_dataframe(new_data, col_id="composition")
    
    # Drop unnecessary columns as done in training
    def clean_data_prediction(df):
        columns_to_drop = [
            'Applied Potential (NH3 Yield)',  'Yield_mg_edited', 'Yield_cm_edited',
            'Elements of electrocatalyst_0', 'Elements of electrocatalyst_1', 
            'Elements of electrocatalyst_2', 'Elements of electrocatalyst_3', 
            'Elements of electrocatalyst_4', 'Elements of electrocatalyst_5', 
            'Elements of electrocatalyst_6', 'Yield_mg', 'Yield_cm',
            'Element_0_ionization_energies', 'Element_1_ionization_energies', 
            'Element_2_ionization_energies', 'Element_3_ionization_energies', 
            'Element_0_electron_configuration', 'Element_1_electron_configuration', 
            'Element_2_electron_configuration', 'Element_3_electron_configuration', 
            'Element_4_ionization_energies', 'Element_4_electron_configuration', 
            'Element_5_ionization_energies', 'Element_5_electron_configuration', 
            'Element_6_ionization_energies', 'Element_6_electron_configuration',
            'pH_nan', 'Electrolyte without concentration_nan', 'N-15 labeling_nan'
        ]
        return df.drop(columns=columns_to_drop, errors='ignore')
    
    new_data = clean_data_prediction(new_data)

    new_data = new_data.drop(columns=['Electrocatalyst'])
    
    # Categorize morphology as done in training
    def categorize_structure_prediction(morphology):
        morphology = morphology.lower()
        if 'nanoparticle' in morphology or 'quantum dots' in morphology or 'nanodots' in morphology: 
            return 'Nanoparticles'
        elif 'nanorod' in morphology:
            return 'Nanorods'
        elif 'nanowire' in morphology:
            return 'Nanowires'
        elif 'nanosheet' in morphology or '2d' in morphology:
            return 'Nanosheets'
        elif 'nanofiber' in morphology:
            return 'Nanofibers'
        elif 'nanocube' in morphology:
            return 'Nanocubes'
        elif 'hollow' in morphology:
            return 'Hollow Structures'
        elif 'porous' in morphology:
            return 'Porous Structures'
        elif 'hybrid' in morphology or 'composite' in morphology:
            return 'Hybrids and Composites'
        elif 'nanotube' in morphology:
            return 'Nanotubes'
        elif 'nanowall' in morphology:
            return 'Nanowalls'
        else:
            return 'Other'
    
    new_data['Structure Type'] = new_data['Morphology of electrocatalyst'].fillna('No values').apply(categorize_structure_prediction)
    
    # One-hot encoding for 'Structure Type'
    new_data = pd.get_dummies(new_data, columns=['Structure Type'])
    
    # Drop original 'Morphology of electrocatalyst'
    new_data = new_data.drop(columns=['Morphology of electrocatalyst'], errors='ignore')
    
    # Final cleanup: drop 'formula' and 'composition'
    new_data = new_data.drop(columns=['formula', 'composition'], axis=1, errors='ignore')
    
    # Convert to float64
    new_data = new_data.astype('float64')
    
    # Remove columns starting with 'Judge' (if any)
    new_data = new_data.loc[:, ~new_data.columns.str.startswith('Judge')]
    
    # Align new_data columns with training data
    for col in features:
        if col not in new_data.columns:
            new_data[col] = 0  # Add missing columns with default value 0
    
    # Ensure the order of columns matches training data
    new_data = new_data[features]
    
    # Scale the features
    new_data_scaled = scaler.transform(new_data)
    
    # Assign clusters using NearestCentroid
    predicted_clusters = nearest_centroid.predict(new_data_scaled)
    
    # Define cluster to FE category mapping
    # Update this mapping based on actual cluster labels
    cluster_to_fe = {3: 'High', 0: 'Low', 1: 'Low', 2: 'Low', 4: 'Low', 5: 'Low'}  # Example mapping
    
    # Map clusters to FE categories
    fe_predictions = [cluster_to_fe.get(cluster, 'Low') for cluster in predicted_clusters]
    
    # Prepare the output DataFrame
    predictions_df = new_data.copy()
    predictions_df['Predicted Cluster'] = predicted_clusters
    predictions_df['Predicted FE Category'] = fe_predictions

    return predictions_df

# Executor setup
executor = LocalCommandLineCodeExecutor(
    work_dir='./input',  # Use the temporary directory to store the code files.
)
CONFIG_FILEPATH = './settings.yaml'

# Initialize Streamlit app
st.title("🤖eNRRCrew")
st.write("Hello! What do you want to know about electrocatalytic NRR?🤗")
# Initialize session state for the toggle
if 'show_content' not in st.session_state:
    st.session_state.show_content = False

def toggle_content():
    st.session_state.show_content = not st.session_state.show_content

# Add the button with a callback function
# st.button("How can I use eNRRCrew?", on_click=toggle_content)
# 定义回调函数
def toggle_content():
    st.session_state.show_content = not st.session_state.get('show_content', False)

# 添加按钮和视频选项
col1, col2 = st.columns(2)

with col1:
    st.button("How can I use eNRRCrew?", on_click=toggle_content)

with col2:
    if st.button("Watch Video Tutorial"):
        st.video("https://youtu.be/8wGIyxSJBFY")

# 显示内容(如果需要)
if st.session_state.get('show_content', False):
    st.write("Here's how eNRRCrew can help you...")


# Display content based on session state
if st.session_state.show_content:
    st.markdown("🚀 **Unlock** eNRR insights by using <span style='color:yellow'>**retrieve**</span> followed by your question!", unsafe_allow_html=True)
    st.markdown("🔮 **Predict the future** with <span style='color:yellow'>**predict yield**</span> for eNRR yield forecasts!", unsafe_allow_html=True)
    st.markdown("⚡ **Boost efficiency** by predicting eNRR's Faradaic efficiency with <span style='color:yellow'>**predict Faradaic efficiency**</span>!", unsafe_allow_html=True)
    st.markdown("📊 **Dive into data** by mentioning <span style='color:yellow'>**CSV**</span> to interact with your CSV files!", unsafe_allow_html=True)
    st.write("Embark on your eNRRCrew journey now! Let's dive in and explore together! 🌊🎯")

    # Initialize session state for input data if not exists
    if 'fe_input_data' not in st.session_state:
        st.session_state.fe_input_data = {}
    if 'yield_input_data' not in st.session_state:
        st.session_state.yield_input_data = {}

    # Faradaic Efficiency Prediction Input
    st.subheader("Input Data for Faradaic Efficiency Prediction")
    with st.expander("Expand to input data for Faradaic Efficiency Prediction"):
        # Define input fields
        input_fields = [
            'Applied Potential (Faraday Efficiency)', 'Electrocatalyst', 
            'Elements of electrocatalyst_0', 'Elements of electrocatalyst_1', 
            'Elements of electrocatalyst_2', 'Elements of electrocatalyst_3', 
            'Elements of electrocatalyst_4', 'Elements of electrocatalyst_5', 
            'Elements of electrocatalyst_6', 'Morphology of electrocatalyst'
        ]

        # Add text inputs for each field
        for field in input_fields:
            st.session_state.fe_input_data[field] = st.text_input(field, key=f"fe_{field}")

        # Add checkboxes for pH and electrolyte
        ph_options = ['acidic', 'alkaline', 'ionic liquid', 'khco3', 'li tfsi', 'nabf4', 'neutral', 'weak acid']
        electrolyte_options = ['h2so4', 'hcl', 'k2so4', 'kclo4', 'koh', 'li2so4', 'licl', 'liclo4', 'lioh', 'na2so4', 'naoh', 'pbs']

        st.subheader("pH")
        for option in ph_options:
            st.session_state.fe_input_data[f'pH_{option}'] = st.checkbox(option, key=f"fe_ph_{option}")

        st.subheader("Electrolyte without concentration")
        for option in electrolyte_options:
            st.session_state.fe_input_data[f'Electrolyte without concentration_{option}'] = st.checkbox(option, key=f"fe_electrolyte_{option}")

        # N-15 labeling
        st.session_state.fe_input_data['N-15 labeling_mentioned'] = st.checkbox("N-15 labeling mentioned", key="fe_n15_labeling")

    if st.button("Submit Data for Faradaic Efficiency Prediction"):
        # Convert the input data to a DataFrame
        df = pd.DataFrame([st.session_state.fe_input_data])
        
        # Save the DataFrame to a CSV file
        df.to_csv('./input/data_for_fe.csv', index=False)
        st.success("Data submitted successfully. You can now use 'predict Faradaic efficiency' in your query.")

    # Yield Prediction Input
    st.subheader("Input Data for Yield Prediction")
    with st.expander("Expand to input data for Yield Prediction"):
        # Define input fields for Yield prediction
        yield_input_fields = [
            'Applied Potential (NH3 Yield)', 'Electrocatalyst', 
            'Elements of electrocatalyst_0', 'Elements of electrocatalyst_1', 
            'Elements of electrocatalyst_2', 'Elements of electrocatalyst_3', 
            'Elements of electrocatalyst_4', 'Elements of electrocatalyst_5', 
            'Elements of electrocatalyst_6', 'Morphology of electrocatalyst'
        ]

        # Add text inputs for each field
        for field in yield_input_fields:
            st.session_state.yield_input_data[field] = st.text_input(field, key=f"yield_{field}")

        # Add checkboxes for pH and electrolyte (same as FE prediction)
        st.subheader("pH")
        for option in ph_options:
            st.session_state.yield_input_data[f'pH_{option}'] = st.checkbox(option, key=f"yield_ph_{option}")

        st.subheader("Electrolyte without concentration")
        for option in electrolyte_options:
            st.session_state.yield_input_data[f'Electrolyte without concentration_{option}'] = st.checkbox(option, key=f"yield_electrolyte_{option}")

        # N-15 labeling
        st.session_state.yield_input_data['N-15 labeling_mentioned'] = st.checkbox("N-15 labeling mentioned", key="yield_n15_labeling")

    if st.button("Submit Data for Yield Prediction"):
        # Convert the input data to a DataFrame
        df = pd.DataFrame([st.session_state.yield_input_data])
        
        # Save the DataFrame to a CSV file
        df.to_csv('./input/data_for_yield.csv', index=False)
        st.success("Data submitted successfully. You can now use 'predict yield' in your query.")

# Session state to hold conversation messages and settings
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'Gen_type' not in st.session_state:
    st.session_state.Gen_type = "single paragraph"
if 'Community' not in st.session_state:
    st.session_state.Community = 0
if 'Search_type' not in st.session_state:
    st.session_state.Search_type = True  # True for local search
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""


st.sidebar.header("Settings")
response_type = st.sidebar.selectbox(
    "(GraphRAG) Content Type",
    ["prioritized list", "single paragraph", "multiple paragraphs", "multiple-page report"],
    index=1
)
community = st.sidebar.slider(
    "(GraphRAG) Community Level",
    min_value=0,
    max_value=2,
    value=0,
    step=1
)
local_search = st.sidebar.checkbox("(GraphRAG) Local Search", value=True)

st.session_state.Gen_type = response_type
st.session_state.Community = community
st.session_state.Search_type = local_search

# User settings
st.sidebar.header("User Configuration")
base_url = st.sidebar.text_input("Base URL", value="https://api.chatanywhere.tech/v1")
api_key = st.sidebar.text_input("API Key", value="", type="password")

# Save the API key and base URL to the .env file
if api_key:
    env_path = ".env"
    set_key(env_path, "GRAPHRAG_API_KEY", api_key)
if base_url:
    env_path = ".env"
    set_key(env_path, "GRAPHRAG_BASE_URL", base_url)

st.session_state['base_url'] = base_url
st.session_state['api_key'] = api_key

if not st.session_state['api_key']:
    st.warning("Please enter your API Key in the sidebar to proceed.")
    st.stop()

llm_config_autogen = {
    "seed": 42,
    "temperature": 0,
    "config_list": [{
        "model": "gpt-4o-mini",
        "base_url": st.session_state['base_url'],
        'api_key': st.session_state['api_key']
    }],
    "timeout": 60000,
}

llm_manager = {
    "seed": 42,
    "temperature": 0,
    "config_list": [{
        "model": "gpt-4o",
        "base_url": st.session_state['base_url'],
        'api_key': st.session_state['api_key']
    }],
    "timeout": 60000,
}

# Initialize agents
retriever = AssistantAgent(
    name="Retriever",
    llm_config=llm_config_autogen,
    system_message="""Only execute the function query_graphRAG to look for context.
                      Output 'TERMINATE' when an answer has been provided.""",
    max_consecutive_auto_reply=1,
    human_input_mode="NEVER",
    description="Retriever Agent"
)

user_proxy = StreamlitUserProxyAgent(
    name="User_Proxy",
    human_input_mode="ALWAYS",
    llm_config=llm_config_autogen,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    system_message='''A human admin. Interact with the retriever, csv_handler to provide any context''',
    description="User Proxy Agent"
)

csv_handler = AssistantAgent(
    name="CSV_Handler",
    human_input_mode="NEVER",
    llm_config=llm_config_autogen,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    system_message='''A CSV handler. You are working with a CSV file named 'data_include_morphology_electrocatalyst.csv'. The column names are: Faraday efficiency, Applied Potential (Faraday Efficiency), Applied Potential (NH3 Yield), Electrocatalyst, Elements of electrocatalyst_0, Elements of electrocatalyst_1, Elements of electrocatalyst_2, Elements of electrocatalyst_3, Elements of electrocatalyst_4, Elements of electrocatalyst_5, Elements of electrocatalyst_6, Morphology of electrocatalyst, and other relevant columns. Answer questions from the user_proxy by writing Python code using the CSV file. Wrap the code in a code block that specifies the script type. Do not include multiple code blocks in one response. Check the execution result returned by the code executor. If there is an error, fix it and output the code again. Always use st.pyplot(plt.gcf()) instead of plt.show(). And always use print function if needed.''',
    description="CSV Handler Agent"
)

code_executor = StreamlitAssistantAgent(
    name="Code_Executor",
    human_input_mode="Always",
    llm_config=llm_config_autogen,
    code_execution_config={
        "executor": executor,
    },
    system_message='''A code executor. You are working with a CSV file named 'data_include_morphology_electrocatalyst.csv'.
        Execute Python code written by the csv_handler and transfer the answer to the user_proxy. Report the result under the context of the question.
        Output 'TERMINATE' when an answer has been provided.''',
    description="Code Executor Agent"
)

yield_predictor = AssistantAgent(
    name="Yield_Predictor",
    llm_config=llm_config_autogen,
    code_execution_config={
        "executor": executor,
    },
    system_message="""Only execute the function predict_yield to predict the yield from the provided information.
                      Output 'TERMINATE' when an answer has been provided.""",
    max_consecutive_auto_reply=1,
    human_input_mode="NEVER",
    description="Yield Predictor Agent"
)

FE_predictor = AssistantAgent(
    name="FE_Predictor",
    llm_config=llm_config_autogen,
    code_execution_config={
        "executor": executor,
    },
    system_message="""Only execute the function predict_fe_high_low to predict the Faraday Efficiency from the provided information.
                      Output 'TERMINATE' when an answer has been provided.""",
    max_consecutive_auto_reply=1,
    human_input_mode="NEVER",
    description="Faraday Efficiency Predictor Agent"
)

# Define a function to handle message sending
def send_message():
    if st.session_state.user_input:
        st.session_state.messages.append(("User", st.session_state.user_input))
        # Handle conversation logic here
        # Implement the conversation flow using the agents

        context = st.session_state.user_input
        INPUT_DIR = None
        ROOT_DIR = '.'
        MAX_ITER = 10

        # Update agents with the latest settings
        RESPONSE_TYPE = st.session_state.Gen_type
        COMMUNITY = st.session_state.Community
        LOCAL_SEARCH = st.session_state.Search_type

        def state_transition(last_speaker, groupchat):
            messages = groupchat.messages

            # 安全获取最后一条消息的内容
            def get_last_message_content():
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, dict):
                        return last_msg.get('content', '').lower()
                    else:
                        return getattr(last_msg, 'content', '').lower()
                return ''

            if last_speaker is user_proxy:
                if len(messages) == 0:
                    return user_proxy

                last_message = get_last_message_content()

                # 使用 elif 确保只匹配一个条件
                if "predict" in last_message and "yield" in last_message:
                    return yield_predictor
                elif "predict" in last_message and "faradaic efficiency" in last_message:
                    return FE_predictor
                elif "csv" in last_message:
                    return csv_handler
                elif "retrieve" in last_message:
                    return retriever
                else:
                    return user_proxy

            elif last_speaker is retriever:
                # 直接返回 user_proxy，无需多余的条件判断
                return user_proxy

            elif last_speaker is csv_handler:
                last_message = get_last_message_content()
                if "```python" in last_message:
                    return code_executor
                else:
                    return csv_handler

            elif last_speaker is code_executor:
                last_message = get_last_message_content()
                if "exitcode: 1" in last_message:
                    return csv_handler
                else:
                    return user_proxy

            elif last_speaker is yield_predictor:
                last_message = get_last_message_content()
                if "exitcode: 1" in last_message:
                    return yield_predictor
                else:
                    return user_proxy

            elif last_speaker is FE_predictor:
                last_message = get_last_message_content()
                if "exitcode: 1" in last_message:
                    return FE_predictor
                else:
                    return user_proxy

            else:
                # 默认返回 user_proxy
                return user_proxy


        # Register functions with agents
        def query_graphRAG(question: Annotated[str, 'Query string containing information that you want from RAG search']) -> str:
            CONFIG_FILEPATH = './settings.yaml'
            if LOCAL_SEARCH:
                result = run_local_search(CONFIG_FILEPATH, INPUT_DIR, ROOT_DIR, COMMUNITY, RESPONSE_TYPE, True, question)[0]
            else:
                result = run_global_search(CONFIG_FILEPATH, INPUT_DIR, ROOT_DIR, COMMUNITY, RESPONSE_TYPE, True, question)[0]
            st.write(result)
            return result

        d_retrieve_content = retriever.register_for_llm(
            description="retrieve content for electrochemical NRR and question answering.", api_style="function"
        )(query_graphRAG)

        for agent in [user_proxy, retriever]:
            agent.register_for_execution()(d_retrieve_content)

        def predict_yield(question: Annotated[str, 'predict yield']) -> str:
            new_data_path = './input/data_for_yield.csv' #data_for_yield.csv
            
            # Check if the file exists
            if not os.path.exists(new_data_path):
                return "Error: No data file found. Please input data for yield prediction first."
            
            try:
                # Read new data
                new_data = pd.read_csv(new_data_path)
                processed_data = preprocess_new_data(new_data)
                all_features = load('./models/features_for_yield_before_RFE.joblib')
                for feature in all_features:
                    if feature not in processed_data.columns:
                        processed_data[feature] = 0.0
                X_all_features = processed_data[all_features]
                scaler = load('./models/scaler_for_yield.joblib')
                X_scaled_all_features = scaler.transform(X_all_features)

                # 读取之前保存的特征选择列表
                with open('./models/selected_features_for_yield.txt', 'r') as f:
                    selected_features = [line.strip() for line in f]
                # 确保所有特征都存在
                X_scaled_selected = pd.DataFrame(X_scaled_all_features, columns=all_features)[selected_features]
                X_scaled = X_scaled_selected
                best_model = load('./models/best_model_for_yield.joblib')
                predictions = best_model.predict(X_scaled)
                probabilities = best_model.predict_proba(X_scaled)[:, 1]
                results_df = pd.DataFrame({
                    'Original_Index': new_data.index,
                    'Predictions': predictions,
                    'Probabilities': probabilities
                })
                final_results = pd.concat([new_data, results_df], axis=1)
                final_results.to_csv('./input/prediction_results_for_yield.csv', index=False)
                st.write("Predicted Results:")
                st.write(final_results)
                return final_results.to_json(orient='records')
            except Exception as e:
                return f"Error during yield prediction: {str(e)}"

        d_predict_yield = yield_predictor.register_for_llm(
            description="predict yield for electrochemical NRR raw data.", api_style="function"
        )(predict_yield)

        for agent in [user_proxy, yield_predictor]:
            agent.register_for_execution()(d_predict_yield)

        def predict_fe(question: Annotated[str, 'predict Faraday Efficiency']) -> str:
            # Predict Faraday Efficiency (implementation needed)
            new_data_path = './input/data_for_fe.csv' # data_for_fe.csv
            model_paths = {
                'scaler': './models/scaler.joblib',
                'nearest_centroid': './models/nearest_centroid.joblib',
                'str_to_comp': './models/str_to_comp.joblib',
                'element_property': './models/element_property.joblib',
                'features': './models/features.joblib'
            }
            predictions_df = predict_fe_high_low(new_data_path, model_paths)
            st.write("Predicted Results:")
            st.write(predictions_df)
            # return predictions_df
            return predictions_df.to_json(orient='records')

        d_predict_fe = FE_predictor.register_for_llm(
            description="predict Faradaic Efficiency for electrochemical NRR raw data.", api_style="function"
        )(predict_fe)

        for agent in [user_proxy, FE_predictor]:
            agent.register_for_execution()(d_predict_fe)

        # Create group chat
        groupchat = autogen.GroupChat(
            agents=[user_proxy, retriever, csv_handler, code_executor, yield_predictor, FE_predictor],
            messages=[],
            max_round=MAX_ITER,
            speaker_selection_method=state_transition, #"auto", # state_transition
            allow_repeat_speaker=True,
        )

        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=llm_manager,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
        )

        
        user_proxy.initiate_chat(manager, message=context)

        #Process messages and display responses
        for message in groupchat.messages:
            if isinstance(message, dict):
                speaker = message.get('role', 'Unknown')
                content = message.get('content', '')
            else:
                speaker = getattr(message, 'sender', 'Unknown')
                content = getattr(message, 'content', '')
            if isinstance(speaker, Agent):
                speaker_name = speaker.name
            else:
                speaker_name = speaker
            st.session_state.messages.append((speaker_name, content))

        # Reset the input field
        st.session_state.user_input = ""

# User input and send button
st.text_input("Your message:", key="user_input")
if st.button("Send", on_click=send_message):
    pass  # The send_message function handles input processing

# Display conversation history
# if st.session_state.messages:
#     st.write("### Conversation History")
#     for speaker, msg in st.session_state.messages:
#         st.write(f"**{speaker}**: {msg}")
