from hyperclovax_few_shot import setup_logger, load_prompts, extract_error_counts, \
    calculate_mqm_score, load_data, save_results, save_checkpoint, visualize_results

from tqdm import tqdm
from openai import OpenAI
from typing import Tuple

import numpy as np
import os
import argparse


# 로거 초기화
logger = setup_logger()

# GEMBA-MQM 프롬프트 생성 함수
def create_gemba_mqm_prompt(source_text: str, translation: str, prompt_path: str) -> str:
    """GEMBA-MQM 논문에서 제시한 Few-shot 프롬프트"""
    prompts = load_prompts(prompt_path)
    return prompts['gemba_mqm_prompt'].format(source_text=source_text, translation=translation)

# EAPrompt 프롬프트 생성 함수
def create_ea_prompt(source_text: str, translation: str, prompt_path: str) -> str:
    """EAPrompt 논문의 프롬프트 형식에 Few-shot 예제를 추가합니다."""
    prompts = load_prompts(prompt_path)
    return prompts['ea_prompt'].format(source_text=source_text, translation=translation)

# GPT 기반 번역 평가 함수
def evaluate_translation_with_gpt(source_text: str, 
                                  translation: str, 
                                  prompt_type: str,
                                  model: str) -> Tuple[str, float]:
    """LLM을 사용하여 번역을 평가하고 MQM 점수를 추출합니다."""
    system_prompt = '''You are an annotator for the quality of machine translation. Your task is to identify
errors and assess the quality of the translation.'''

    # 프롬프트 생성
    if prompt_type == 'gemba_mqm':
        user_prompt = create_gemba_mqm_prompt(source_text, translation, args.prompt_path)
    elif prompt_type == 'ea_prompt':
        user_prompt = create_ea_prompt(source_text, translation, args.prompt_path)
    
    # LLM 호출
    try:
        # OpenAI API 클라이언트 초기화
        client = OpenAI()
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

        logger.info(f"\n{prompt_type} 방식으로 평가 시작")
        
        # 각 번역에 대해 LLM 평가 수행
        for i, row in tqdm(df.iterrows(), total=len(df)):
            # LLM 평가 수행
            response, mqm_score = evaluate_translation_with_gpt(
                source_text=row['Source'],
                translation=row['MT'],
                prompt_type=prompt_type,
                model=args.model,
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

        # 최종 결과 저장
        corr_result = save_results(df, args.output_dir, prompt_type, args.model)
        all_corr_results.append(corr_result)
    
    # 결과 시각화
    visualize_results(all_corr_results, args.output_dir)

    logger.info("\n평가 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM을 사용한 번역 품질 평가 (논문 원본 프롬프트 적용)')
    parser.add_argument('--data_path', type=str, required=True, help='평가 데이터셋 경로')
    parser.add_argument('--output_dir', type=str, default='results/gpt4o', help='결과 저장 디렉토리')
    parser.add_argument('--model', type=str, default='gpt-4o', choices=['gpt-4o', 'gpt-4.1'], help='사용할 LLM 모델')
    parser.add_argument('--prompt_path', type=str, required=True, help='프롬프트 파일 경로')
    parser.add_argument('--save_interval', type=int, default=50, help='중간 결과 저장 간격')
    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)



# python gpt-4o_few_shot.py --data_path data/03_03_Createll_241111_firsttrans_eval_he.csv --model gpt-4o --prompt_path prompts/few_shot_prompt.yaml
# export OPENAI_API_KEY=''
