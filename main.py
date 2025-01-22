import pandas as pd
import re
import os
from llm_setting import create_relevance_checker

base_dir = 'Task_9'
csv_file_name = '2024_11_25_20_37_22.csv'
csv_path = os.path.join(base_dir, csv_file_name)

def save_results(results, result_df, filename='filtered_results'):
    # 创建paper到LLM评估结果的映射
    llm_responses = {item1.strip(): item2.strip() for item1, item2 in results}
    # 过滤掉不合法数据
    filtered_results = [
        re.sub(r"```", "", item1).strip()
        for item1, _ in results
        if re.search(r"[a-zA-Z]", item1)
    ]

    # 将过滤后的结果保存到文件
    with open(f'{filename}.txt', 'w', encoding='utf-8') as f:
        for result in filtered_results:
            f.write(result + '\n')

    # 过滤 result_df 中 paper 列的值等于 filtered_results 的行
    filtered_result_df = result_df[result_df['paper'].isin(filtered_results)]

    # 添加LLM列
    filtered_result_df['LLM'] = filtered_result_df['paper'].map(llm_responses)

    # 按照 rating 列从大到小排序
    filtered_result_df = filtered_result_df.sort_values(by='rating', ascending=False)

    # 保存成 CSV 文件
    filtered_result_df.to_csv(f'{filename}.csv', index=False)
    return filtered_results, filtered_result_df

def read_csv(csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 检查是否存在 'Rating' 列
    if 'Rating' in df.columns:
        # 统计按照rating过滤
        score_df = df.groupby('paper').agg(
            Rating_Mean=('Rating', 'mean'),
            Rating_min=('Rating', 'min'),
            Rating_max=('Rating', 'max')
        ).reset_index()

        accept_paper = score_df[(score_df['Rating_Mean'] >= 5.5) & (score_df['Rating_Mean'] <= 15.5)][['paper', 'Rating_Mean']].apply(tuple, axis=1).tolist()
    else:
        df['Rating'] = 15
        accept_paper = df[['paper', 'Rating']].apply(tuple, axis=1).tolist()

    # 得到聚合后的结果
    result_df = pd.DataFrame()
    if 'paper' not in df.columns:
        raise ValueError("The column 'paper' is not in the DataFrame.")
    if 'Abstract' not in df.columns:
        raise ValueError("The column 'Abstract' is not in the DataFrame.")
    if 'Keywords' not in df.columns:
        df['Keywords'] = ''
    # 遍历may_accept_paper中的每个值
    for value, rating in accept_paper:
        # 获取对应值的所有行
        rows = df[df['paper'] == value]
        
        # 获取"paper", "abstract", "keywords"列的值（假设这些列的值是相同的）
        # 创建一个包含"paper", "abstract", "keywords"列的新DataFrame
        paper = rows['paper'].iloc[0]
        abstract = rows['Abstract'].iloc[0]
        keywords = rows['Keywords'].iloc[0]
        new_row = {
            'paper': paper,
            'abstract': abstract,
            'keywords': keywords,
            'rating': rating
        }
        
        # 是否有'Summary'列
        if 'Summary' in df.columns:
            # 获取所有的"summary"列的值
            summaries = rows['Summary'].tolist()
            
            # 增加多个"summary"列的新DataFrame
            for i, summary in enumerate(summaries):
                new_row[f'summary{i+1}'] = summary
        
        # 将新行添加到结果DataFrame中
        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
    return result_df

def create_llm_input(result_df):
    # 创建传给llm的df
    # 创建新的DataFrame
    info_df = pd.DataFrame(columns=['paper', 'info'])
    summary_columns = [col for col in result_df.columns if 'summary' in col]
    # 遍历result_df的每一行
    for _, row in result_df.iterrows():
        # 获取基本信息
        paper = row['paper']
        keywords = row['keywords']
        abstract = row['abstract']
        
        # 构建基础信息
        info = f"""
paper: {paper}
keywords: {keywords}
abstract: {abstract}
"""
        # 获取并检查summary
        summaries = []
        for idx, summary_key in enumerate(summary_columns): 
            if summary_key in row and pd.notna(row[summary_key]):  # 检查summary是否存在且有效
                summaries.append(f"reviewer summary{idx}: {row[summary_key]}")
        # 将所有信息组合
        info += "\n".join(summaries)
        # 添加到新DataFrame
        info_df = pd.concat([info_df, pd.DataFrame({'paper': [paper], 'info': [info]})], 
                        ignore_index=True)
    return info_df

def run_llm(csv_path, model='qwen2.5-coder:32b', core_setting="3DGS"):
    result_df = read_csv(csv_path)
    info_df = create_llm_input(result_df)
    results = []

    check_relevance, role_setting = create_relevance_checker(core_setting, model)

    print(f"""
model: {model}
**role_setting**
{role_setting}
"""
    )

    for _, row in info_df.iterrows():
        relevance_response = check_relevance(row['info'])
        if re.search(r'[a-zA-Z]', relevance_response) and "空字符串" not in relevance_response:
            relevance_response = re.sub(r'```', '', relevance_response).strip()
            paper_name = row['paper']
            print(paper_name)
            results.append((paper_name, relevance_response))

    os.makedirs(os.path.join(base_dir, 'output'), exist_ok=True)
    save_results(results, result_df, filename=os.path.join(base_dir, 'output', 'llm_results'))

core_setting = ['光影编辑', '3DGS relighting']
model = 'qwen2.5-coder:32b'
run_llm(csv_path, model, core_setting)