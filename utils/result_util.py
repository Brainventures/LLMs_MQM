from utils.model_util import extract_error_counts
from utils.logger_util import setup_logger

from scipy.stats import kendalltau
from typing import List, Dict, Any

import os
import json
import pandas as pd
import logging

logger = logging.getLogger()

# 결과 저장 함수
def save_results(df: pd.DataFrame, output_dir: str, prompt_type: str, model_name: str):
    """최종 평가 결과 저장"""
    # 결과 파일 경로
    results_path = os.path.join(output_dir, f"{prompt_type}_{model_name.replace('/', '_')}_fewshot_results.csv")
    df.to_csv(results_path, index=False)
    
    # 상관관계 계산 및 저장
    corr, p_value = kendalltau(df['MQMScore'], df[f'LLM_MQM_{prompt_type}'])
    
    corr_results = {
        'prompt_type': prompt_type,
        'model': model_name,
        'kendall_tau': corr,
        'p_value': p_value,
        'sample_size': len(df)
    }
    
    corr_path = os.path.join(output_dir, f"{prompt_type}_{model_name.replace('/', '_')}_fewshot_correlation.json")
    with open(corr_path, 'w') as f:
        json.dump(corr_results, f, indent=4)
    
    logger.info(f"{prompt_type} + {model_name} 평가 결과:")
    logger.info(f"Kendall's Tau: {corr:.4f} (p-value: {p_value:.4f})")
    
    return corr_results

# 중간 결과 저장 함수
def save_json_checkpoint(results: List, output_dir: str, prompt_type: str, model_name: str, iteration: int):
    """중간 평가 결과를 JSON 형식으로 저장"""
    checkpoint_path = os.path.join(output_dir, f"{prompt_type}_{model_name.replace('/', '_')}_checkpoint_{iteration}.json")
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(f"중간 결과가 {checkpoint_path}에 저장되었습니다.")

def process_prompt_results(response: str, mqm_score: int, prompt_type: str, model: str) -> Dict[str, Any]:
    """각 프롬프트 유형별 평가 결과 처리"""
    result = {
        "prompt_type": prompt_type,
        "model": model,
        f"LLM_Response_{prompt_type}": response,
        f"LLM_Score_{prompt_type}": mqm_score,
    }

    if prompt_type == 'gemba_mqm':
        critical, major, minor = extract_error_counts(response, prompt_type)
        result.update({
            "gemba_mqm_critical": critical,
            "gemba_mqm_major": major,
            "gemba_mqm_minor": minor
        })

    elif prompt_type == 'ea_prompt':
        major, minor = extract_error_counts(response, prompt_type)
        result.update({
            "ea_prompt_major": major,
            "ea_prompt_minor": minor
        })

    return result
    
# 최종 결과 저장 함수
def save_json_final(results: List, output_dir: str, model_name: str):
    """최종 평가 결과를 JSON 형식으로 저장"""
    result_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_few_shot_response.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(f"최종 결과가 {result_path}에 저장되었습니다.")
