from scipy.stats import kendalltau
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import torch
import yaml
import logging


# 로깅 설정
def setup_logger():
    """로깅 설정 초기화"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# 로거 초기화
logger = setup_logger()

# 프롬프트 로드 함수
def load_prompts(prompt_path):
    """YAML 파일에서 프롬프트 로드"""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        return prompts
    except Exception as e:
        logger.error(f"프롬프트 파일 로드 중 오류 발생: {e}")

def load_model(args):
    # HyperCLOVAX 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    logger.info(f"{args.model} 모델 로드 완료!")

    return tokenizer, model

# GEMBA-MQM 프롬프트 생성 함수
def create_gemba_mqm_prompt(source_text: str, translation: str, prompt_path: str) -> str:
    """GEMBA-MQM 논문에서 제시한 Few-shot 프롬프트를 HyperCLOVAX 형식에 맞게 조정"""
    prompts = load_prompts(prompt_path)
    prompt_content = prompts['gemba_mqm_prompt'].format(source_text=source_text, translation=translation)
    
    # HyperCLOVAX 모델에 맞는 프롬프트 형식 적용
    formatted_prompt = f"<|user|>\n{prompt_content}\n<|assistant|>"
    return formatted_prompt

# EAPrompt 프롬프트 생성 함수
def create_ea_prompt(source_text: str, translation: str, prompt_path: str) -> str:
    """EAPrompt 논문의 프롬프트 형식에 Few-shot 예제를 추가하고 HyperCLOVAX 형식에 맞게 조정"""
    prompts = load_prompts(prompt_path)
    prompt_content = prompts['ea_prompt'].format(source_text=source_text, translation=translation)
    
    # HyperCLOVAX 모델에 맞는 프롬프트 형식 적용
    formatted_prompt = f"<|user|>\n{prompt_content}\n<|assistant|>"
    return formatted_prompt

# HyperCLOVAX 모델을 사용하여 번역 평가 함수
def evaluate_translation_with_hyperclovax(source_text: str, 
                              translation: str, 
                              prompt_type: str) -> Tuple[str, float]:
    """HyperCLOVAX 모델을 사용하여 번역을 평가하고 MQM 점수 추출"""
    system_prompt = '''You are an annotator for the quality of machine translation. Your task is to identify
errors and assess the quality of the translation.'''

    # 프롬프트 생성
    if prompt_type == 'gemba_mqm':
        user_prompt = create_gemba_mqm_prompt(source_text, translation, args.prompt_path)
    elif prompt_type == 'ea_prompt':
        user_prompt = create_ea_prompt(source_text, translation, args.prompt_path)

    # HyperCLOVAX 모델 호출
    try:
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=2048, 
                stop_strings=["<|endofturn|>", "<|stop|>"], 
                tokenizer=tokenizer,
                do_sample=False,
                temperature=0.0,
                top_p=1.0)
        
        # 응답 텍스트 추출 (토크나이저로 디코딩)
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # HyperCLOVAX 응답에서 <|assistant|> 이후 부분만 추출
        response_text = full_output.split("<|assistant|>")[-1].strip()
        
        # 에러 카운트 추출 및 MQM 점수 계산
        if prompt_type == 'gemba_mqm':
            error_counts = extract_error_counts(response_text, prompt_type)
            mqm_score = calculate_mqm_score(error_counts, prompt_type)
        elif prompt_type == 'ea_prompt':
            error_counts = extract_error_counts(response_text, prompt_type)
            mqm_score = calculate_mqm_score(error_counts, prompt_type)
        
        return response_text, mqm_score
    
    except Exception as e:
        logger.error(f"{args.model} 호출 중 오류 발생: {e}")
        return str(e), None

def extract_error_counts(response_text: str, prompt_type: str) -> tuple:
    """
    모델 응답에서 에러 카운트(critical, major, minor)를 추출
    """
    if prompt_type == 'gemba_mqm':
        # 'Output 3 numbers ONLY with the format: "x, x, x"' 질문 찾기
        question_pattern = r'Output 3 numbers ONLY with the format: "x, x, x"'
        question_match = re.search(question_pattern, response_text)
        
        if question_match:
            # 질문 이후의 텍스트에서 답변 찾기
            remaining_text = response_text[question_match.end():]
            # A: 다음에 "x, x, x" 형태로 나오는 패턴 찾기
            answer_pattern = r'A:\s*(\d+),\s*(\d+),\s*(\d+)'
            answer_match = re.search(answer_pattern, remaining_text)
            
            if answer_match:
                critical = int(answer_match.group(1))
                major = int(answer_match.group(2))
                minor = int(answer_match.group(3))
                return critical, major, minor
            
        else:
            # 응답 전체에서 "x, x, x" 패턴 찾기
            for line in response_text.strip().split('\n'):
                match = re.search(r'(\d+),\s*(\d+),\s*(\d+)', line)
                if match:
                    critical = int(match.group(1))
                    major = int(match.group(2))
                    minor = int(match.group(3))
                    return critical, major, minor
            
    elif prompt_type == 'ea_prompt':
        # 'Output 2 numbers ONLY with the format: "x, x"' 질문 찾기
        question_pattern = r'Output 2 numbers ONLY with the format: "x, x"'
        question_match = re.search(question_pattern, response_text)
        
        if question_match:
            # 질문 이후의 텍스트에서 답변 찾기
            remaining_text = response_text[question_match.end():]
            # A: 다음에 "x, x" 형태로 나오는 패턴 찾기
            answer_pattern = r'A:\s*(\d+),\s*(\d+)'
            answer_match = re.search(answer_pattern, remaining_text)
            
            if answer_match:
                major = int(answer_match.group(1))
                minor = int(answer_match.group(2))
                return major, minor

        else:
            # 응답 전체에서 "x, x" 패턴 찾기
            for line in response_text.strip().split('\n'):
                match = re.search(r'(\d+),\s*(\d+)', line)
                if match:
                    major = int(match.group(1))
                    minor = int(match.group(2))
                    return major, minor

def calculate_mqm_score(error_counts: tuple, prompt_type: str) -> float:
    """에러 카운트를 기반으로 MQM 점수 계산"""
    if prompt_type == 'gemba_mqm':
        critical, major, minor = error_counts
        if any(count is None for count in [critical, major, minor]):
            return None
        mqm_score = 100 - (25 * critical + 5 * major + 1 * minor)
    elif prompt_type == 'ea_prompt':
        major, minor = error_counts
        if any(count is None for count in [major, minor]):
            return None
        mqm_score = 100 - (10 * major + 2 * minor)
    else:
        logger.error(f"알 수 없는 프롬프트 유형: {prompt_type}")
        return None
    
    # MQM 점수가 음수가 될 수 있으므로, 여기서 최소값을 0으로 제한
    mqm_score = max(0, mqm_score)
    
    return mqm_score

# 데이터 로드 및 전처리 함수
def load_data(data_path: str) -> pd.DataFrame:
    """번역 평가 데이터셋을 로드하고 전처리"""
    df = pd.read_csv(data_path)
    df = df[['Source', 'MT', 'MTPE', 'MQMScore']]
    
    # 필요한 컬럼 확인
    required_cols = ['Source', 'MT', 'MTPE', 'MQMScore']
    for col in required_cols:
        if col not in df.columns:
            error_msg = f"데이터셋에 필요한 컬럼 '{col}'이 없습니다."
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    return df

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
def save_checkpoint(df: pd.DataFrame, output_dir: str, prompt_type: str, model_name: str, iteration: int):
    """중간 평가 결과 저장"""
    checkpoint_path = os.path.join(output_dir, f"{prompt_type}_{model_name.replace('/', '_')}_checkpoint_{iteration}.csv")
    df.to_csv(checkpoint_path, index=False)
    logger.info(f"중간 결과가 {checkpoint_path}에 저장되었습니다.")

# 시각화 함수
def visualize_results(corr_results: List[Dict], output_dir: str):
    """평가 결과 시각화"""
    df_corr = pd.DataFrame(corr_results)
    
    # 프롬프트 유형별 바 플롯 생성
    plt.figure(figsize=(10, 6))
    sns.barplot(x='prompt_type', y='kendall_tau', data=df_corr)
    plt.title(f"Correlation with Human Evaluation by Evaluation Method using {args.model}")
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Kendall's Tau")
    plt.axhline(y=0.5, color='r', linestyle='--', label='Target correlation coefficient (0.5)')
    plt.legend()
    plt.tight_layout()
    
    # 결과 저장
    plt.savefig(os.path.join(output_dir, f'correlation_{args.model.replace("/", "_")}.png'))
    plt.close()

def main(args):
    # 데이터 로드
    logger.info(f"데이터셋 로드 중: {args.data_path}")
    df = load_data(args.data_path)
    logger.info(f"로드된 샘플 수: {len(df)}")

    # 결과 저장할 리스트
    all_corr_results = []
    
    # 각 프롬프트 유형에 대해 평가 수행
    for prompt_type in ['gemba_mqm', 'ea_prompt']:
        # 결과 컬럼 추가
        response_col = f'LLM_Response_{prompt_type}'
        score_col = f'LLM_MQM_{prompt_type}'
        
        # 에러 카운트 저장을 위한 컬럼 추가
        if prompt_type == 'gemba_mqm':
            df['Critical_Errors'] = np.nan
            df['Major_Errors'] = np.nan
            df['Minor_Errors'] = np.nan
        elif prompt_type == 'ea_prompt':
            df['EA_Major_Errors'] = np.nan
            df['EA_Minor_Errors'] = np.nan
        
        df[response_col] = None
        df[score_col] = np.nan
        
        logger.info(f"{prompt_type} 방식으로 평가 시작")
        
        # 각 번역에 대해 LLM 평가 수행
        for i, row in tqdm(df.iterrows(), total=len(df)):
            # LLM 평가 수행
            response, mqm_score = evaluate_translation_with_hyperclovax(
                source_text=row['Source'],
                translation=row['MT'],
                prompt_type=prompt_type
            )
            
            # 결과 저장
            df.at[i, response_col] = response
            
            # 에러 카운트 추출 및 저장
            if prompt_type == 'gemba_mqm':
                critical, major, minor = extract_error_counts(response, prompt_type)
                if critical is not None:
                    df.at[i, 'Critical_Errors'] = critical
                    df.at[i, 'Major_Errors'] = major
                    df.at[i, 'Minor_Errors'] = minor
            elif prompt_type == 'ea_prompt':
                major, minor = extract_error_counts(response, prompt_type)
                if major is not None:
                    df.at[i, 'EA_Major_Errors'] = major
                    df.at[i, 'EA_Minor_Errors'] = minor
            
            # MQM 점수 저장
            if mqm_score is not None:
                df.at[i, score_col] = mqm_score
            
            # 중간 결과 저장
            if (i + 1) % args.save_interval == 0:
                save_checkpoint(df, args.output_dir, prompt_type, args.model, i + 1)
                logger.info(f"{i+1}/{len(df)} 샘플 처리 완료")
                
            # 메모리 캐시 정리
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        
        # 최종 결과 저장
        corr_result = save_results(df, args.output_dir, prompt_type, args.model)
        all_corr_results.append(corr_result)
    
    # 결과 시각화
    visualize_results(all_corr_results, args.output_dir)
    
    logger.info("평가 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HyperCLOVAX LLM을 사용한 번역 품질 평가 (논문 원본 프롬프트 적용)')
    parser.add_argument('--data_path', type=str, required=True, help='평가 데이터셋 경로')
    parser.add_argument('--output_dir', type=str, default='results/hyperclovax', help='결과 저장 디렉토리')
    parser.add_argument('--model', type=str, default='naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B', help='사용할 HyperCLOVAX 모델')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--prompt_path', type=str, required=True, help='프롬프트 파일 경로')
    parser.add_argument('--save_interval', type=int, default=50, help='중간 결과 저장 간격')
    args = parser.parse_args()

    tokenizer, model = load_model(args)

    main(args)



# python hyperclovax_few_shot.py --data_path data/03_03_Createll_241111_firsttrans_eval_he.csv --prompt_path prompts/ko_few_shot_prompt.yaml
