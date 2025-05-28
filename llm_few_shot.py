from utils.logger_util import setup_logger
from utils.data_util import load_data
from utils.result_util import save_results, process_prompt_results, save_json_final, save_json_checkpoint
from utils.model_util import extract_error_counts, calculate_mqm_score, load_model
from utils.prompt_util import create_ea_prompt, create_gemba_mqm_prompt

from tqdm import tqdm
from typing import Tuple

import os
import argparse
import torch
import numpy as np


# 로거 초기화
logger = setup_logger()

# 평가자 모델을 사용하여 번역 평가 함수
def evaluate_translation_with_qwen(source_text: str,
                                   reference_text: str,
                                   translation: str, 
                                   prompt_type: str) -> Tuple[str, float]:
    """LLM을 평가자로 활용하여 번역을 평가하고 MQM 점수 추출"""
    system_prompt = '''You are an assistant that provides concise, focused answers. Do not include reasoning steps or use the <think> tag in your responses'''

    # 프롬프트 생성
    if prompt_type == 'gemba_mqm':
        user_prompt = create_gemba_mqm_prompt(source_text, translation, reference_text, args.prompt_path)
    elif prompt_type == 'ea_prompt':
        user_prompt = create_ea_prompt(source_text, translation, reference_text, args.prompt_path)
    
    # 모델 호출
    try:
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            tokenize=True,
            return_tensors="pt").to(args.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=1024,
                do_sample=False,
            )

        response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        response_text = response_text.split("assistant")[-1].strip()

        logger.info(f"LLM 응답 결과:\n{response_text}")

        # 에러 카운트 추출 및 MQM 점수 계산
        if prompt_type == 'gemba_mqm':
            error_counts = extract_error_counts(response_text, prompt_type)
            mqm_score = calculate_mqm_score(error_counts, prompt_type)
        elif prompt_type == 'ea_prompt':
            error_counts = extract_error_counts(response_text, prompt_type)
            mqm_score = calculate_mqm_score(error_counts, prompt_type)

        logger.info(f"LLM 점수 결과: {mqm_score}")
        
        return response_text, mqm_score
    
    except Exception as e:
        logger.error(f"{args.model} 호출 중 오류 발생: {e}")
        return str(e), None

def main(args):
    # 데이터 로드
    logger.info(f"데이터셋 로드 중: {args.data_path}")
    df = load_data(args.data_path)
    logger.info(f"로드된 샘플 수: {len(df)}")
    
    # 결과 저장할 리스트 초기화
    all_corr_results = []
    all_json_results = []
    
# 각 프롬프트 유형에 대해 평가 수행
    for prompt_type in ['gemba_mqm', 'ea_prompt']:
        # 결과 컬럼 추가
        score_col = f'LLM_MQM_{prompt_type}'
        df[score_col] = np.nan

        logger.info(f"\n{prompt_type} 방식으로 평가 시작")
        
        # 각 번역에 대해 LLM 평가 수행
        for i, row in tqdm(df.iterrows(), total=len(df)):
            # LLM 평가 수행
            response, mqm_score = evaluate_translation_with_qwen(
                source_text=row['Source'],
                translation=row['MT'],
                reference_text=row['MTPE'],
                prompt_type=prompt_type,
            )
            
            # MQM 점수 저장
            if mqm_score is not None:
                df.at[i, score_col] = mqm_score
            
            all_json_results.append(process_prompt_results(response, mqm_score, prompt_type, args.model))

            # 중간 결과 저장
            if (i + 1) % args.save_interval == 0:
                save_json_checkpoint(all_json_results, args.output_dir, prompt_type, args.model, i + 1)
                logger.info(f"{i+1}/{len(df)} 샘플 처리 완료")

        corr_result = save_results(df, args.output_dir, prompt_type, args.model)
        all_corr_results.append(corr_result)

    save_json_final(all_json_results, args.output_dir, args.model)
    
    logger.info("\n평가 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM을 사용한 번역 품질 평가 (논문 원본 프롬프트 적용)')
    parser.add_argument('--data_path', type=str, default='data/03_03_Createll_241111_firsttrans_eval_he.csv', help='평가 데이터셋 경로')
    parser.add_argument('--output_dir', type=str, required=True, help='결과 저장 디렉토리')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--prompt_path', type=str, required=True, help='프롬프트 파일 경로')
    parser.add_argument('--save_interval', type=int, default=250, help='중간 결과 저장 간격')
    parser.add_argument('--hf_token', type=str, help='Hugging Face API 토큰')
    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # Hugging Face 토큰 설정
    os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    tokenizer, model = load_model(args)

    if args.model == 'google/gemma-3-12b-it':
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    main(args)



# python llm_few_shot.py --prompt_path prompts/Version3/few_shot_prompt.yaml --model google/gemma-3-12b-it --output_dir results/gemma3/Version3 --hf_token 
# python llm_few_shot.py --prompt_path prompts/Version3/few_shot_prompt.yaml --model meta-llama/Llama-3.1-8B-Instruct --output_dir results/llama3.1/Version3 --hf_token
# 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
