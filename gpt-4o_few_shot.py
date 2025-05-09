import pandas as pd
import numpy as np
import re
import os
from scipy.stats import kendalltau
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from typing import List, Dict, Tuple
import json
import argparse
import yaml

# 명령행 인자 파서 설정
parser = argparse.ArgumentParser(description='LLM을 사용한 번역 품질 평가 (논문 원본 프롬프트 적용)')
parser.add_argument('--data_path', type=str, required=True, help='평가 데이터셋 경로')
parser.add_argument('--output_dir', type=str, default='results', help='결과 저장 디렉토리')
parser.add_argument('--model', type=str, default='gpt-4o', choices=['gpt-4o', 'gpt-4.1'], help='사용할 LLM 모델')
parser.add_argument('--few_shot_count', type=int, default=3, help='Few-shot 예제 개수 (0-5)')
parser.add_argument('--prompt_path', type=str, required=True, help='프롬프트 파일 경로')
# parser.add_argument('--n_samples', type=int, default=250, help='평가할 샘플 수')
# parser.add_argument('--save_interval', type=int, default=100, help='중간 결과 저장 간격')
args = parser.parse_args()

# 출력 디렉토리 생성
os.makedirs(args.output_dir, exist_ok=True)

# OpenAI API 클라이언트 초기화
client = OpenAI()

# 프롬프트 로드 함수
def load_prompts(prompt_path):
    """YAML 파일에서 프롬프트 정의를 로드합니다."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        return prompts
    except Exception as e:
        print(f"프롬프트 파일 로드 중 오류 발생: {e}")

# 프롬프트 로드
print(f"프롬프트 파일 로드 중: {args.prompt_path}")
prompts = load_prompts(args.prompt_path)
print("프롬프트 로드 완료!")

# GEMBA-MQM 프롬프트 생성 함수
def create_gemba_mqm_prompt(source_text: str, translation: str) -> str:
    """GEMBA-MQM 논문에서 제시한 Few-shot 프롬프트"""
    return prompts['gemba_mqm_prompt'].format(source_text=source_text, translation=translation)

# EAPrompt 프롬프트 생성 함수
def create_ea_prompt(source_text: str, translation: str) -> str:
    """EAPrompt 논문의 프롬프트 형식에 Few-shot 예제를 추가합니다."""
    return prompts['ea_prompt'].format(source_text=source_text, translation=translation)

# LLM을 사용하여 번역 평가 함수 (Few-shot 파라미터 추가)
def evaluate_translation_with_llm(source_text: str, 
                                  translation: str, 
                                  prompt_type: str,
                                  model: str) -> Tuple[str, float]:
    """LLM을 사용하여 번역을 평가하고 MQM 점수를 추출합니다."""
    
    system_prompt = '''You are a professional translator and linguistics expert specialized in English-Korean translation.
  Your task is to evaluate the quality of machine translations based on specific criteria.'''

    # 프롬프트 생성
    if prompt_type == 'gemba_mqm':
        user_prompt = create_gemba_mqm_prompt(source_text, translation)
    elif prompt_type == 'ea_prompt':
        user_prompt = create_ea_prompt(source_text, translation)
    else:
        raise ValueError("지원되지 않는 프롬프트 유형입니다. 'gemba_mqm' 또는 'ea_prompt'를 사용하세요.")
    
    # LLM 호출
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=1024,
        )
        
        response_text = response.choices[0].message.content
        
        # MQM 점수 추출
        mqm_score = extract_mqm_score(response_text)
        
        return response_text, mqm_score
    
    except Exception as e:
        print(f"LLM 호출 중 오류 발생: {e}")
        return str(e), None

# MQM 점수 추출 함수
def extract_mqm_score(response_text: str) -> float:
    """LLM 응답에서 MQM 점수를 추출합니다."""
    
    # 마지막 숫자 찾기 (최종 MQM 점수일 가능성이 높음)
    numbers = re.findall(r'\d+\.?\d*', response_text)
    if numbers:
        return float(numbers[-1])
    
    # "MQM score" 또는 "score" 근처에서 숫자 찾기
    mqm_score_match = re.search(r'MQM\s*(?:score|점수)?\s*[=:]?\s*(\d+\.?\d*)', response_text, re.IGNORECASE)
    if mqm_score_match:
        return float(mqm_score_match.group(1))
    
    # "final score" 근처에서 숫자 찾기
    final_score_match = re.search(r'final\s*(?:score|점수)?\s*[=:]?\s*(\d+\.?\d*)', response_text, re.IGNORECASE)
    if final_score_match:
        return float(final_score_match.group(1))
    
    return None

# 데이터 로드 및 전처리 함수
def load_data(data_path: str) -> pd.DataFrame:
    """번역 평가 데이터셋을 로드하고 전처리합니다."""
    
    df = pd.read_csv(data_path)
    df = df[['Source', 'MT', 'MTPE', 'MQMScore']]
    
    # 필요한 컬럼 확인
    required_cols = ['Source', 'MT', 'MTPE', 'MQMScore']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"데이터셋에 필요한 컬럼 '{col}'이 없습니다.")
    
    # 샘플 수 제한
    # if n_samples and n_samples < len(df):
    #     df = df.sample(n_samples, random_state=42)
    
    return df

# 결과 저장 함수 (Few-shot 정보 추가)
def save_results(df: pd.DataFrame, output_dir: str, prompt_type: str, model: str, num_examples: int):
    """평가 결과를 저장합니다."""
    
    # 결과 파일 경로
    results_path = os.path.join(output_dir, f"{prompt_type}_{model}_fewshot{num_examples}_results.csv")
    df.to_csv(results_path, index=False)
    
    # 상관관계 계산 및 저장
    corr, p_value = kendalltau(df['MQMScore'], df[f'LLM_MQM_{prompt_type}'])
    
    corr_results = {
        'prompt_type': prompt_type,
        'model': model,
        'few_shot_count': num_examples,
        'kendall_tau': corr,
        'p_value': p_value,
        'sample_size': len(df)
    }
    
    corr_path = os.path.join(output_dir, f"{prompt_type}_{model}_fewshot{num_examples}_correlation.json")
    with open(corr_path, 'w') as f:
        json.dump(corr_results, f, indent=4)
    
    print(f"\n{prompt_type} + {model} (Few-shot: {num_examples}) 평가 결과:")
    print(f"Kendall's Tau: {corr:.4f} (p-value: {p_value:.4f})")
    
    return corr_results

# 중간 결과 저장 함수 (Few-shot 정보 추가)
def save_checkpoint(df: pd.DataFrame, output_dir: str, prompt_type: str, model: str, num_examples: int, iteration: int):
    """중간 평가 결과를 저장합니다."""
    
    checkpoint_path = os.path.join(output_dir, f"{prompt_type}_{model}_fewshot{num_examples}_checkpoint_{iteration}.csv")
    df.to_csv(checkpoint_path, index=False)

# 시각화 함수 (Few-shot 정보 추가)
def visualize_results(corr_results: List[Dict], output_dir: str):
    """평가 결과를 시각화합니다."""
    
    df_corr = pd.DataFrame(corr_results)
    
    # Few-shot 개수별 바 플롯 생성
    plt.figure(figsize=(12, 7))
    sns.barplot(x='prompt_type', y='kendall_tau', hue='few_shot_count', data=df_corr)
    plt.title("Correlation with Human Evaluation by Few-shot Prompting Evaluation Method (Kendall's Tau)")
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Kendall's Tau")
    plt.axhline(y=0.5, color='r', linestyle='--', label='Target correlation coefficient (0.5)')
    plt.legend(title='Few-shot')
    plt.tight_layout()
    
    # 결과 저장
    plt.savefig(os.path.join(output_dir, 'correlation_fewshot_comparison.png'))
    plt.close()
    
    # 모델별 바 플롯 생성
    if len(df_corr['model'].unique()) > 1:
        plt.figure(figsize=(12, 7))
        sns.barplot(x='prompt_type', y='kendall_tau', hue='model', data=df_corr)
        plt.title("Correlation with Human Evaluation by LLM Evaluation Method (Kendall's Tau)")
        plt.xlabel("Evaluation Metric")
        plt.ylabel("Kendall's Tau")
        plt.axhline(y=0.5, color='r', linestyle='--', label='Target correlation coefficient (0.5)')
        plt.legend(title='Model')
        plt.tight_layout()
        
        # 결과 저장
        plt.savefig(os.path.join(output_dir, 'correlation_model_comparison.png'))
        plt.close()

def main():
    # 데이터 로드
    print(f"데이터셋 로드 중: {args.data_path}")
    df = load_data(args.data_path)
    print(f"로드된 샘플 수: {len(df)}")
    
    # 결과 저장할 리스트
    all_corr_results = []
    
    # 각 프롬프트 유형에 대해 평가 수행
    for prompt_type in ['gemba_mqm', 'ea_prompt']:
        # 결과 컬럼 추가
        response_col = f'LLM_Response_{prompt_type}'
        score_col = f'LLM_MQM_{prompt_type}'
        
        df[response_col] = None
        df[score_col] = np.nan
        
        print(f"\n{prompt_type} 방식으로 평가 시작")
        
        # 각 번역에 대해 LLM 평가 수행
        for i, row in tqdm(df.iterrows(), total=len(df)):
            # LLM 평가 수행
            response, mqm_score = evaluate_translation_with_llm(
                source_text=row['Source'],
                translation=row['MT'],
                prompt_type=prompt_type,
                model=args.model,
            )
            
            # 결과 저장
            df.at[i, response_col] = response
            if mqm_score is not None:
                df.at[i, score_col] = mqm_score
            
            # 중간 결과 저장
            # if (i + 1) % args.save_interval == 0:
            #     save_checkpoint(df, args.output_dir, prompt_type, args.model, args.few_shot_count, i + 1)
                
            #     # API 호출 제한 회피를 위한 대기
            #     time.sleep(2)
        
        # 최종 결과 저장
        corr_result = save_results(df, args.output_dir, prompt_type, args.model, args.few_shot_count)
        all_corr_results.append(corr_result)
    
    # 결과 시각화
    visualize_results(all_corr_results, args.output_dir)
    
    print("\n평가 완료!")

if __name__ == "__main__":
    main()


# python gpt-4o_few_shot.py --data_path data/03_03_Createll_241111_firsttrans_eval_he.csv --model gpt-4o --prompt_path prompts/few_shot_prompt.yaml
# export OPENAI_API_KEY=''
