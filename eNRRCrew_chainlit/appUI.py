import autogen
from rich import print
import chainlit as cl
from typing_extensions import Annotated
from chainlit.input_widget import (
   Select, Slider, Switch)
from autogen import AssistantAgent, UserProxyAgent
from utils.chainlit_agents import ChainlitUserProxyAgent, ChainlitAssistantAgent
from graphrag.query.cli import run_global_search, run_local_search
from autogen.coding import DockerCommandLineCodeExecutor
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
import csv
import json
import re
from openai import OpenAI


# Setup OpenAI API client
client = OpenAI(
    api_key="your_api_key_here",
    base_url="your_api_base_url_here"
)

def extract_data_from_text(text):
    """
    Use LLM to extract relevant information from text and generate structured data
    """
    # Use a stricter prompt to ensure correct JSON format return
    prompt = f"""
    Extract information from the following text and return it in strict JSON format with the following fields:

    {{
      "Applied_Potential": "electrode potential value, only keep the numerical values, units not needed. Note that negative numbers should retain their negative sign.",
      "Electrocatalyst": "complete name of the catalyst",
      "Elements_of_electrocatalyst": ["element1", "element2", "element3", null, null, null, null], Please extract all the elements involved, including the active center and the support, Use abbreviations.
      "Morphology_of_electrocatalyst": "morphology description of the catalyst",
      "pH_values": {{
        "ph_acidic": false or true,
        "ph_alkaline": false or true,
        "ph_ionic liquid": false or true,
        "ph_khco3": false or true,
        "ph_li tfsi": false or true,
        "ph_nabf4": false or true,
        "ph_neutral": false or true,
        "ph_weak acid": false or true
      }},
      "Electrolyte_values": {{
        "h2so4": false or true,
        "hcl": false or true,
        "k2so4": false or true,
        "kclo4": false or true,
        "koh": false or true,
        "li2so4": false or true,
        "licl": false or true,
        "liclo4": false or true,
        "lioh": false or true,
        "na2so4": false or true,
        "naoh": false or true,
        "pbs": false or true
      }},
      "N15_labeling_mentioned": false or true
    }}

    Please strictly follow the format above and return only the valid JSON object.
    Do not add any additional explanations or text.
    Use double quotes instead of single quotes.
    Fill in the actual values based on the text; keep unmentioned values as null or false.

    Text to analyze:
    {text}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional data extraction assistant. Your task is to extract structured data from text and return it in strict JSON format without any additional text or explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Use very low temperature for consistent formatted output
            response_format={"type": "json_object"}  # Explicitly request JSON format
        )
        
        # Get response content
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print("API returned content:", content)
            
            # Try using regex to extract JSON part
            import re
            json_pattern = re.compile(r'({[\s\S]*})')
            match = json_pattern.search(content)
            
            if match:
                try:
                    result = json.loads(match.group(1))
                    return result
                except:
                    pass
            
            # If still unable to parse, try fixing common JSON errors
            fixed_content = content.replace("'", '"')  # Replace single quotes with double quotes
            fixed_content = re.sub(r'(\w+):', r'"\1":', fixed_content)  # Add quotes to key names
            
            try:
                result = json.loads(fixed_content)
                return result
            except:
                pass
                
            # If all attempts fail, ask user to manually adjust the prompt and retry
            raise ValueError("Unable to parse JSON from API response. Please modify the prompt to get a proper JSON format response.")
            
    except Exception as e:
        print(f"API call or parsing error: {e}")
        raise

def create_csv_for_fe(data, filename="./input/updated_challenge_data_real_for_fe.csv"):
    """
    Create a CSV file for Faraday Efficiency prediction based on the extracted data
    """
    # Define CSV headers
    headers = [
        "Applied Potential (Faraday Efficiency)",
        "Electrocatalyst",
    ]
    
    # Add element-related headers
    for i in range(7):
        headers.append(f"Elements of electrocatalyst_{i}")
    
    headers.append("Morphology of electrocatalyst")
    
    # Add pH-related headers
    ph_types = [
        "pH_acidic", "pH_alkaline", "pH_ionic liquid", "pH_khco3", 
        "pH_li tfsi", "pH_nabf4", "pH_neutral", "pH_weak acid"
    ]
    headers.extend(ph_types)
    
    # Add electrolyte-related headers
    electrolyte_types = [
        "Electrolyte without concentration_h2so4", "Electrolyte without concentration_hcl",
        "Electrolyte without concentration_k2so4", "Electrolyte without concentration_kclo4",
        "Electrolyte without concentration_koh", "Electrolyte without concentration_li2so4",
        "Electrolyte without concentration_licl", "Electrolyte without concentration_liclo4",
        "Electrolyte without concentration_lioh", "Electrolyte without concentration_na2so4",
        "Electrolyte without concentration_naoh", "Electrolyte without concentration_pbs"
    ]
    headers.extend(electrolyte_types)
    
    # Add N-15 labeling header
    headers.append("N-15 labeling_mentioned")
    
    # Create row data
    row = {}
    
    # Fill in basic data
    row["Applied Potential (Faraday Efficiency)"] = data.get("Applied_Potential", "")
    row["Electrocatalyst"] = data.get("Electrocatalyst", "")
    
    # Fill in element data
    elements = data.get("Elements_of_electrocatalyst", [])
    for i in range(7):
        if i < len(elements) and elements[i] is not None:
            row[f"Elements of electrocatalyst_{i}"] = elements[i]
        else:
            row[f"Elements of electrocatalyst_{i}"] = ""
    
    # Fill in morphology data
    row["Morphology of electrocatalyst"] = data.get("Morphology_of_electrocatalyst", "")
    
    # Fill in pH data
    ph_values = data.get("pH_values", {})
    for ph_type in ph_types:
        key = ph_type.lower()
        row[ph_type] = "TRUE" if ph_values.get(key.replace("ph_", ""), False) else "FALSE"
    
    # Fill in electrolyte data
    electrolyte_values = data.get("Electrolyte_values", {})
    for electrolyte_type in electrolyte_types:
        key = electrolyte_type.replace("Electrolyte without concentration_", "").lower()
        row[electrolyte_type] = "TRUE" if electrolyte_values.get(key, False) else "FALSE"
    
    # Fill in N-15 labeling data
    row["N-15 labeling_mentioned"] = "TRUE" if data.get("N15_labeling_mentioned", False) else "FALSE"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write to CSV file
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)
    
    print(f"CSV file for FE prediction successfully created: {filename}")
    return filename

def create_csv_for_yield(data, filename="./input/updated_challenge_data_real_for_yield.csv"):
    """
    Create a CSV file for NH3 Yield prediction based on the extracted data
    """
    # Define CSV headers
    headers = [
        "Applied Potential (NH3 Yield)",
        "Electrocatalyst",
    ]
    
    # Add element-related headers
    for i in range(7):
        headers.append(f"Elements of electrocatalyst_{i}")
    
    headers.append("Morphology of electrocatalyst")
    
    # Add pH-related headers
    ph_types = [
        "pH_acidic", "pH_alkaline", "pH_ionic liquid", "pH_khco3", 
        "pH_li tfsi", "pH_nabf4", "pH_neutral", "pH_weak acid"
    ]
    headers.extend(ph_types)
    
    # Add electrolyte-related headers
    electrolyte_types = [
        "Electrolyte without concentration_h2so4", "Electrolyte without concentration_hcl",
        "Electrolyte without concentration_k2so4", "Electrolyte without concentration_kclo4",
        "Electrolyte without concentration_koh", "Electrolyte without concentration_li2so4",
        "Electrolyte without concentration_licl", "Electrolyte without concentration_liclo4",
        "Electrolyte without concentration_lioh", "Electrolyte without concentration_na2so4",
        "Electrolyte without concentration_naoh", "Electrolyte without concentration_pbs"
    ]
    headers.extend(electrolyte_types)
    
    # Add N-15 labeling header
    headers.append("N-15 labeling_mentioned")
    
    # Create row data
    row = {}
    
    # Fill in basic data
    row["Applied Potential (NH3 Yield)"] = data.get("Applied_Potential", "")
    row["Electrocatalyst"] = data.get("Electrocatalyst", "")
    
    # Fill in element data
    elements = data.get("Elements_of_electrocatalyst", [])
    for i in range(7):
        if i < len(elements) and elements[i] is not None:
            row[f"Elements of electrocatalyst_{i}"] = elements[i]
        else:
            row[f"Elements of electrocatalyst_{i}"] = ""
    
    # Fill in morphology data
    row["Morphology of electrocatalyst"] = data.get("Morphology_of_electrocatalyst", "")
    
    # Fill in pH data
    ph_values = data.get("pH_values", {})
    for ph_type in ph_types:
        key = ph_type.lower()
        row[ph_type] = "TRUE" if ph_values.get(key.replace("ph_", ""), False) else "FALSE"
    
    # Fill in electrolyte data
    electrolyte_values = data.get("Electrolyte_values", {})
    for electrolyte_type in electrolyte_types:
        key = electrolyte_type.replace("Electrolyte without concentration_", "").lower()
        row[electrolyte_type] = "TRUE" if electrolyte_values.get(key, False) else "FALSE"
    
    # Fill in N-15 labeling data
    row["N-15 labeling_mentioned"] = "TRUE" if data.get("N15_labeling_mentioned", False) else "FALSE"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write to CSV file
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)
    
    print(f"CSV file for yield prediction successfully created: {filename}")
    return filename


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
    #new_data = new_data.astype('float64')
    
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


executor = LocalCommandLineCodeExecutor(
    work_dir='./input',  # Use the temporary directory to store the code files.
    )

CONFIG_FILEPATH = './settings.yaml'

llm_config_autogen = {
    "seed": 42,  # change the seed for different trials
    "temperature": 0,
    "config_list": [{"model": "gpt-4.1-mini", 
                     "base_url": "your_api_base_url_here", 
                     'api_key': 'your_api_key_here'},
    ],
    "timeout": 60,
}

llm_manager = {
    "seed": 42,  # change the seed for different trials
    "temperature": 0,
    "config_list": [{"model": "gpt-4.1", 
                     "base_url": "your_api_base_url_here", 
                     'api_key': 'your_api_key_here'},
    ],
    "timeout": 60,
}


@cl.on_chat_start
async def on_chat_start():
  try:
    settings = await cl.ChatSettings(
            [      
                Switch(id="Search_type", label="(GraphRAG) Local Search", initial=True),       
                Select(
                    id="Gen_type",
                    label="(GraphRAG) Content Type",
                    values=["prioritized list", "single paragraph", "multiple paragraphs", "multiple-page report"],
                    initial_index=1,
                ),          
                Slider(
                    id="Community",
                    label="(GraphRAG) Community Level",
                    initial=0,
                    min=0,
                    max=2,
                    step=1,
                ),

            ]
        ).send()

    response_type = settings["Gen_type"]
    community = settings["Community"]
    local_search = settings["Search_type"]
    
    cl.user_session.set("Gen_type", response_type)
    cl.user_session.set("Community", community)
    cl.user_session.set("Search_type", local_search)

    retriever = AssistantAgent(
       name="Retriever", 
       llm_config=llm_config_autogen, 
       system_message="""Retriever Agent""",
       max_consecutive_auto_reply=1,
       human_input_mode="NEVER", 
       description="""Retriever Agent. Only execute the function query_graphRAG to look for nitrogen reduction reaction related context. 
                    Output 'TERMINATE' when an answer has been provided.""" # To help the Group Chat Manager select the next agent, we also set the description of the agents. Without the description, the Group Chat Manager will use the agents’ system_message, which may be not be the best choice.
    )

    user_proxy = ChainlitUserProxyAgent(
        name="User_Proxy",
        human_input_mode="ALWAYS",
        llm_config=llm_config_autogen,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        system_message='''A human admin''',
        description="User Proxy Agent, A human admin. Interact with the retriever, csv_handler, yield_predictor and FE_predictor to provide any context"
    )
    
    csv_handler = AssistantAgent(
        name="CSV_Handler",
        human_input_mode="NEVER",
        llm_config=llm_config_autogen,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        system_message='''CSV Handler Agent''',
        description='''CSV Handler Agent, A CSV handler. You are working with a csv file in data_include_morphology_electrocatalyst.csv, the column names are: Faraday efficiency,
            Applied Potential (Faraday Efficiency), Applied Potential (NH3 Yield), Electrocatalyst, Elements of electrocatalyst_0, Elements of electrocatalyst_1, Elements of electrocatalyst_2,
            Elements of electrocatalyst_3, Elements of electrocatalyst_4, Elements of electrocatalyst_5, Elements of electrocatalyst_6, Morphology of electrocatalyst, Judge_Nanoparticles,
            Judge_3D, Judge_2D, Judge_1D, Judge_Urchin, Judge_SAC, Judge_Hollow, Judge_Core_Shell, Judge_alloy, Judge_Functionalized, Judge_heterostructures, Judge_COF, 
            Judge_MOF, Judge_Hydroxide_Oxyhydroxide, Judge_Spinel, Judge_Perovskite, Judge_Prussian_Blue, Judge_Amorphous, Judge_Ultrasound, Judge_c3n4,
            Judge_MXene, Judge_Steel, Judge_Carbon_Cloth, Judge_Hydrothermal, Judge_Spray, Judge_Phys_Chem_Deposition, Judge_Electrochemical_Deposition,
            pH_acidic, pH_alkaline, pH_ionic liquid, pH_khco3, pH_li tfsi, pH_nabf4, pH_neutral, pH_weak acid, pH_nan, Electrolyte without concentration_h2so4,
            Electrolyte without concentration_hcl, Electrolyte without concentration_k2so4, Electrolyte without concentration_kclo4, Electrolyte without concentration_koh,
            Electrolyte without concentration_li2so4, Electrolyte without concentration_licl, Electrolyte without concentration_liclo4, Electrolyte without concentration_lioh,
            Electrolyte without concentration_na2so4, Electrolyte without concentration_naoh, Electrolyte without concentration_pbs, Electrolyte without concentration_nan,
            Yield_mg_edited, N-15 labeling_mentioned, Yield_cm_edited, N-15 labeling_nan, Yield_mg, Yield_cm, and extra columns.
            answering questions from the user_proxy by writing Python code with the CSV file. Wrap the code in a code block that specifies the script type. 
            The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended 
            to be executed by the code executor. Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. 
            Check the execution result returned by the code executor. If the result indicates there is an error, fix the error and 
            output the code again. Suggest the full code instead of partial code or code changes. Use print statements to output the result if the output is a number.
            If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, 
            revisit your assumption, collect additional info you need, and think of a different approach to try.'''
    )

    code_executor = ChainlitAssistantAgent(
        name="Code_Executor",
        human_input_mode="NEVER",
        llm_config=llm_config_autogen,
        code_execution_config={
            "executor": executor,
        },
        system_message='''A code executor''',
        description='''Code Executor Agent. A code executor. You are working with a csv file in data_include_morphology_electrocatalyst.csv.
        Execute Python code written by the csv_handler and transfer the answer to the user_proxy and report the result under the context of the question.
        Output 'TERMINATE' when an answer has been provided. The Code output should copy to user_proxy.'''
    )

    yield_predictor = AssistantAgent(
        name="Yield_Predictor", 
        llm_config=llm_config_autogen, 
        code_execution_config={
            "executor": executor,
        },
        system_message="""Yield Predictor Agent.
        Your goal is to predict the NH3 yield.
        1. If the user provides text describing an electrocatalyst system for yield prediction (e.g., "predict yield for ...[text description]..."), you MUST first call the `process_text_to_csv` function with that text. This function will extract data and create the necessary CSV file.
        2. After `process_text_to_csv` has successfully run (you will see its output), or if the user simply asks to "predict yield" without new text (implying the CSV is already prepared from a previous step or is static), you MUST then call the `predict_yield` function to get the prediction.
        Do not ask the user to upload information if they have already provided it in their text. The `process_text_to_csv` function handles the conversion from text to the required CSV format. Output 'TERMINATE' after providing the prediction.""",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER", 
        description="""Yield Predictor Agent: Processes text for catalyst info if provided, then predicts NH3 yield using the generated/updated CSV."""
    )

    FE_predictor = AssistantAgent(
        name="FE_Predictor",
        llm_config=llm_config_autogen,
        code_execution_config={
            "executor": executor,
        },
        system_message="""Faraday Efficiency Predictor Agent.
        Your goal is to predict the Faradaic Efficiency (FE).
        1. If the user provides text describing an electrocatalyst system for FE prediction (e.g., "predict FE for ...[text description]..."), you MUST first call the `process_text_to_csv` function with that text. This function will extract data and create the necessary CSV file.
        2. After `process_text_to_csv` has successfully run (you will see its output), or if the user simply asks to "predict FE" without new text (implying the CSV is already prepared), you MUST then call the `predict_fe` function to get the prediction.
        Do not ask the user to upload information if they have already provided it in their text. The `process_text_to_csv` function handles the conversion from text to the required CSV format.
        Output 'TERMINATE' after providing the prediction.""",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        description="""Faraday Efficiency Predictor Agent: Processes text for catalyst info if provided, then predicts Faradaic Efficiency using the generated/updated CSV."""
    )

    cl.user_session.set("Query Agent", user_proxy)
    cl.user_session.set("Retriever", retriever)
    cl.user_session.set("CSV Handler Agent", csv_handler)
    cl.user_session.set("Code_Executor", code_executor)
    cl.user_session.set("Yield_Predictor", yield_predictor)
    cl.user_session.set("FE_Predictor", FE_predictor)

    msg = cl.Message(content=f"""Hello! What do you want to know about electrochemical NRR?     
                     """, 
                     author="User_Proxy")
    await msg.send()
    
  except Exception as e:
    print("Error: ", e)
    pass

@cl.on_settings_update
async def setup_agent(settings):
    response_type = settings["Gen_type"]
    community = settings["Community"]
    local_search = settings["Search_type"]
    cl.user_session.set("Gen_type", response_type)
    cl.user_session.set("Community", community)
    cl.user_session.set("Search_type", local_search)
    print("on_settings_update", settings)

@cl.on_message
async def run_conversation(message: cl.Message):
    print("Running conversation")
    INPUT_DIR = None
    ROOT_DIR = '.'    
    CONTEXT = message.content
    MAX_ITER = 20   
    RESPONSE_TYPE = cl.user_session.get("Gen_type")
    COMMUNITY = cl.user_session.get("Community")
    LOCAL_SEARCH = cl.user_session.get("Search_type")

    retriever = cl.user_session.get("Retriever")
    user_proxy = cl.user_session.get("Query Agent")
    csv_handler = cl.user_session.get("CSV Handler Agent")
    code_executor = cl.user_session.get("Code_Executor")
    yield_predictor = cl.user_session.get("Yield_Predictor")
    FE_predictor = cl.user_session.get("FE_Predictor")

    print("Setting groupchat")

    def state_transition(last_speaker, groupchat):
        """Define a customized speaker selection function for the agent setup.

        Returns:
            Return an `Agent` class or a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
        """
        messages = groupchat.messages

        # if len(messages) <= 1:
        #     # Start the conversation with the User_Proxy
        #     return user_proxy

        if last_speaker is user_proxy:
            # The user proxy interacts with both retriever and csv_handler
            if "csv" in messages[-1]["content"].lower():
                # If the message is related to retrieval, let the Retriever speak
                return csv_handler
            # if "predict" in messages[-1]["content"].lower():
            if "predict" in messages[-1]["content"].lower() and "yield" in messages[-1]["content"].lower():
                return yield_predictor
            if "predict" in messages[-1]["content"].lower() and "faradaic efficiency" in messages[-1]["content"].lower():
                return FE_predictor
            else:
                return retriever

        elif last_speaker is retriever:
            # After the retriever provides context, allow the User_Proxy to continue
            if messages[-1]["content"].lower() not in ['math_expert','physics_expert']:
                return user_proxy
            else:
                return user_proxy

        elif last_speaker is csv_handler:
            # After executing CSV-related tasks, return control to the User_Proxy
            if "```python" in messages[-1]["content"]:
                return code_executor
            else:
                return csv_handler
            
        elif last_speaker is code_executor:
            if "exitcode: 1" in messages[-1]["content"]:
                return csv_handler
            else:
                return user_proxy
        
        elif last_speaker is yield_predictor:
            if "exitcode: 1" in messages[-1]["content"]:
                return yield_predictor
            else:
                return user_proxy
        
        elif last_speaker is FE_predictor:
            if "exitcode: 1" in messages[-1]["content"]:
                return FE_predictor
            else:
                return user_proxy
           
        else:
            pass
            return None

    async def query_graphRAG(
          question: Annotated[str, 'Query string containing information that you want from RAG search']
                          ) -> str:
        CONFIG_FILEPATH = './settings.yaml'
        if LOCAL_SEARCH:
            result = run_local_search(CONFIG_FILEPATH, INPUT_DIR, ROOT_DIR, COMMUNITY, RESPONSE_TYPE, True, question)[0]
        else:
            result = run_global_search(CONFIG_FILEPATH, INPUT_DIR, ROOT_DIR, COMMUNITY, RESPONSE_TYPE, True, question)[0]
        await cl.Message(content=result).send()
        return result

    for caller in [user_proxy, retriever]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve content for electrochemical NRR and question answering.", api_style="function"
        )(query_graphRAG)

    for agents in [user_proxy, retriever]:
        agents.register_for_execution()(d_retrieve_content)

    async def process_text_to_csv(
        text: Annotated[str, 'Text description containing information about electrocatalysts']
     ) -> dict:
        """Process text to extract data and create CSV files for prediction"""
        try:
            # Extract structured data from text
            extracted_data = extract_data_from_text(text)
            
            # Create CSV files for both FE and yield prediction
            fe_csv_path = create_csv_for_fe(extracted_data)
            yield_csv_path = create_csv_for_yield(extracted_data)
            
            result = {
                "fe_csv_path": fe_csv_path,
                "yield_csv_path": yield_csv_path,
                "extracted_data": extracted_data
            }
            
            # Send a message with the extracted information
            msg = f"Successfully extracted data from text:\n\n"
            msg += f"**Applied Potential**: {extracted_data.get('Applied_Potential', 'Not found')}\n"
            msg += f"**Electrocatalyst**: {extracted_data.get('Electrocatalyst', 'Not found')}\n"
            
            elements = extracted_data.get('Elements_of_electrocatalyst', [])
            elements_str = ", ".join([str(e) for e in elements if e is not None])
            msg += f"**Elements**: {elements_str}\n"
            
            msg += f"**Morphology**: {extracted_data.get('Morphology_of_electrocatalyst', 'Not found')}\n"
            
            # Add pH information
            ph_values = extracted_data.get('pH_values', {})
            ph_types = [k for k, v in ph_values.items() if v]
            if ph_types:
                msg += f"**pH types**: {', '.join(ph_types)}\n"
            else:
                msg += "**pH types**: None specified\n"
            
            # Add electrolyte information
            electrolyte_values = extracted_data.get('Electrolyte_values', {})
            electrolytes = [k for k, v in electrolyte_values.items() if v]
            if electrolytes:
                msg += f"**Electrolytes**: {', '.join(electrolytes)}\n"
            else:
                msg += "**Electrolytes**: None specified\n"
            
            # Add N15 labeling information
            msg += f"**N15 labeling mentioned**: {extracted_data.get('N15_labeling_mentioned', False)}\n"
            
            msg += f"\nCreated CSV files for prediction:\n- {fe_csv_path}\n- {yield_csv_path}"
            
            await cl.Message(content=msg).send()
            result = {
                "status": "success", # Add a status
                "message": "Data extracted and CSV files created successfully.",
                "fe_csv_path": fe_csv_path,
                "yield_csv_path": yield_csv_path,
                "extracted_data": extracted_data
            }
            return result
            
        except Exception as e:
            error_msg = f"Error processing text: {str(e)}"
            await cl.Message(content=error_msg).send()
            return {"error": error_msg}

    for caller in [user_proxy, yield_predictor, FE_predictor]:
        d_process_text = caller.register_for_llm(
            description="Process text description about electrocatalystic system and create CSV files for prediction.", 
            api_style="function"
        )(process_text_to_csv)

    for agents in [user_proxy, yield_predictor, FE_predictor]:
        agents.register_for_execution()(d_process_text)

    async def predict_yield() -> dict:
        # Read new data
        new_data = pd.read_csv('./input/updated_challenge_data_real_for_yield.csv')
        
        # Preprocess data
        processed_data = preprocess_new_data(new_data)
        all_features = load('./models/features_for_yield_before_RFE.joblib')
        for feature in all_features:
            if feature not in processed_data.columns:
                processed_data[feature] = 0.0
        X_all_features = processed_data[all_features]
        scaler = load('./models/scaler_for_yield.joblib')
        X_scaled_all_features = scaler.transform(X_all_features)
        with open('./models/selected_features_for_yield.txt', 'r') as f:
            selected_features = [line.strip() for line in f]
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
        
        # Merge original data and prediction results
        final_results = pd.concat([new_data, results_df], axis=1)
        
        # Save results
        final_results.to_csv('prediction_results.csv', index=False)
        
        print("Predicted Results:")
        print(final_results)
        
        # Prepare the output in Markdown table format
        output = "| Electrocatalyst | Predictions |\n"
        output += "|----------------|-------------|\n"
        for i, row in final_results.iterrows():
            output += f"| {row['Electrocatalyst']} | {row['Predictions']} |\n"
        
        await cl.Message(content=output).send()
        
        # Return a dictionary with the results
        return {
            'message': 'Yield prediction completed successfully.', # Added a success message
            'electrocatalyst_predictions': dict(zip(final_results['Electrocatalyst'], final_results['Predictions']))
        }

    for caller in [yield_predictor]: # Only yield_predictor needs this specific predict_yield
        d_predict_yield = caller.register_for_llm(
            description="Predicts NH3 yield based on the data in './input/updated_challenge_data_real_for_yield.csv'. This CSV file should have been generated or updated by the `process_text_to_csv` function if new experimental data was provided by the user in text form.", 
            api_style="function"
        )(predict_yield)

    for agents in [yield_predictor]: # Only yield_predictor needs this specific predict_yield
        agents.register_for_execution()(d_predict_yield)
        
    async def predict_fe() -> dict:
        # Load the new data and model paths
        new_data_path = './input/updated_challenge_data_real_for_fe.csv'
        model_paths = {
            'scaler': './models/scaler.joblib',
            'nearest_centroid': './models/nearest_centroid.joblib',
            'str_to_comp': './models/str_to_comp.joblib',
            'element_property': './models/element_property.joblib',
            'features': './models/features.joblib'
        }
        
        # Load new data
        new_data = pd.read_csv(new_data_path)
        
        # Predict Faraday Efficiency
        predictions_df = predict_fe_high_low(new_data_path, model_paths)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Original_Index': new_data.index,
            'Predicted_Cluster': predictions_df['Predicted Cluster'],
            'Predicted FE Category': predictions_df['Predicted FE Category']
        })
        
        # Merge original data with predictions
        final_results = pd.concat([new_data, results_df], axis=1)
        
        # Save the predictions
        final_results.to_csv('prediction_fe_results.csv', index=False)

        print("Predicted Results:")
        print(final_results)
        
        # Prepare the output in Markdown table format
        output = "| Electrocatalyst | Predicted FE Category |\n"
        output += "|----------------|---------------------|\n"
        for i, row in final_results.iterrows():
            output += f"| {row['Electrocatalyst']} | {row['Predicted FE Category']} |\n"
        
        await cl.Message(content=output).send()
        
        # Return a dictionary with the results
        return {
            'message': 'Faradaic Efficiency prediction completed successfully.', # Added a success message
            'electrocatalyst_fe_predictions': dict(zip(predictions_df['Electrocatalyst'], predictions_df['Predicted FE Category']))
        }

    for caller in [FE_predictor]:
        d_predict_fe = caller.register_for_llm(
            description="Predicts Faradaic Efficiency (FE) category (High/Low) based on the data in './input/updated_challenge_data_real_for_fe.csv'. This CSV file should have been generated or updated by the `process_text_to_csv` function if new experimental data was provided by the user in text form.", 
            api_style="function"
        )(predict_fe) # api_style="function"

    for agents in [FE_predictor]:
        agents.register_for_execution()(d_predict_fe)



    groupchat = autogen.GroupChat(
        agents=[user_proxy, retriever, csv_handler, code_executor, yield_predictor, FE_predictor],
        messages=[],
        max_round=MAX_ITER,
        speaker_selection_method="auto", # state_transition
        allow_repeat_speaker=True,
        send_introductions=True, # In the previous example, we set the description of the agents to help the Group Chat Manager select the next agent. This only helps the Group Chat Manager, however, does not help the participating agents to know about each other. Sometimes it is useful have each agent introduce themselves to other agents in the group chat. This can be done by setting the send_introductions=True.
    )
    manager = autogen.GroupChatManager(groupchat=groupchat,
                                      llm_config=llm_manager, 
                                      is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
                                      code_execution_config=False,
                                      system_message='''A group chat manager. Handle the conversation between the user_proxy, retriever, csv_handler, code_executor, yield_predictor, and FE_predictor according to the user_proxy's message.
                                      You decide which agent to reply. For general questions about NRR, involve the retriever. Output 'TERMINATE' when an answer has been provided.''',
                                      )    


    # Conversation Logic. Edit to change your first message based on the Task you want to get done # 
    try:    
        if len(groupchat.messages) == 0: 
            await cl.make_async(user_proxy.initiate_chat)( manager, message=CONTEXT, )
        elif len(groupchat.messages) < MAX_ITER:
            await cl.make_async(user_proxy.send)( manager, message=CONTEXT, )
        elif len(groupchat.messages) == MAX_ITER:  
            await cl.make_async(user_proxy.send)( manager, message="exit", )
    except Exception as e:
        error_msg = f"Error during conversation: {str(e)}"
        print(error_msg)
        await cl.Message(content=error_msg).send()
