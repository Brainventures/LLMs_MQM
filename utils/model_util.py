from utils.data_util import setup_logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import re
import logging

logger = logging.getLogger()

def load_model(args):
    # HyperCLOVAX 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    logger.info(f"{args.model} 모델 로드 완료!")

    return tokenizer, model

def extract_error_counts(response_text: str, prompt_type: str) -> tuple:
    """
    모델 응답에서 에러 카운트(critical, major, minor)를 추출
    """
    if prompt_type == 'gemba_mqm':
        # 응답 중 마지막 문장에서 "x, x, x" 패턴 찾기
        last_line = response_text.strip().split('\n')[-1]  # 마지막 줄만 추출
        match = re.search(r'(\d+),\s*(\d+),\s*(\d+)', last_line)
        if match:
            critical = int(match.group(1))
            major = int(match.group(2))
            minor = int(match.group(3))
            return critical, major, minor

    elif prompt_type == 'ea_prompt':
        # 응답 중 마지막 문장에서 "x, x, x" 패턴 찾기
        last_line = response_text.strip().split('\n')[-1]  # 마지막 줄만 추출
        match = re.search(r'(\d+),\s*(\d+)', last_line)
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
    
    # MQM 점수가 음수가 될 수 있으므로, 여기서 최소값을 0으로 제한
    mqm_score = max(0, mqm_score)
    
    return mqm_score