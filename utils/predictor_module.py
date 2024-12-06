import pandas as pd
import numpy as np
import random
from joblib import load
from sklearn.preprocessing import StandardScaler
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

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
    new_data = clean_data(new_data)

    # 处理形态学特征等
    new_data = handle_structure_features(new_data)

    # 读取之前保存的特征选择列表
    with open('./selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f]
    
    # 处理缺失特征
    for feature in selected_features:
        if feature not in new_data.columns:
            new_data[feature] = 0  # 填充默认值
    
    # 确保只包含选定的特征
    new_data = new_data[selected_features]
    
    return new_data

def clean_data(df):
    # 清理不需要的列
    df = df.drop(columns=[
        'Faraday efficiency', 'Applied Potential (Faraday Efficiency)',
        'Elements of electrocatalyst_0', 'Elements of electrocatalyst_1',
        'Elements of electrocatalyst_2', 'Elements of electrocatalyst_3',
        'Elements of electrocatalyst_4', 'Elements of electrocatalyst_5',
        'Elements of electrocatalyst_6', 'Yield_mg', 'Yield_cm',
        # 其他不需要的列
    ], errors='ignore')
    return df

def handle_structure_features(new_data):
    # 处理形态学特征
    def categorize_structure(morphology):
        if pd.isna(morphology):
            return 'No values'
        morphology = str(morphology).lower()
        if 'nanoparticle' in morphology:
            return 'Nanoparticles'
        # 其他分类...
        return 'Other'

    # 添加结构类型
    new_data['Structure Type'] = new_data['Morphology of electrocatalyst'].apply(categorize_structure)
    new_data = pd.get_dummies(new_data, columns=['Structure Type'])
    new_data = new_data.drop(columns=['Morphology of electrocatalyst'], errors='ignore')

    return new_data

# def predict_yield(new_data_path):
def predict_yield():
    # 读取新数据
    new_data = pd.read_csv('new_data.csv')
    
    # 预处理数据
    processed_data = preprocess_new_data(new_data)
    
    # 读取之前保存的特征选择列表
    with open('./selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f]
    
    # 确保所有选定的特征都存在
    for feature in selected_features:
        if feature not in processed_data.columns:
            processed_data[feature] = 0.0
    
    # 仅保留选定的特征
    X = processed_data[selected_features]
    
    # 手动标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 加载最佳模型并预测
    best_model = load('best_model.joblib')
    
    # 预测
    predictions = best_model.predict(X_scaled)
    probabilities = best_model.predict_proba(X_scaled)[:, 1]
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Original_Index': new_data.index,
        'Predictions': predictions,
        'Probabilities': probabilities
    })
    
    # 合并原始数据和预测结果
    final_results = pd.concat([new_data, results_df], axis=1)
    
    # 保存结果
    final_results.to_csv('prediction_results.csv', index=False)
    
    print("预测结果:")
    print(final_results)
    
    return final_results
