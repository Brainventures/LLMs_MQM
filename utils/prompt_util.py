from utils.data_util import setup_logger
import yaml
import logging

logger = logging.getLogger()

# 프롬프트 로드 함수
def load_prompts(prompt_path):
    """YAML 파일에서 프롬프트 로드"""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        return prompts
    except Exception as e:
        logger.error(f"프롬프트 파일 로드 중 오류 발생: {e}")

# GEMBA-MQM 프롬프트 생성 함수
def create_gemba_mqm_prompt(source_text: str, translation: str, prompt_path: str) -> str:
    """GEMBA-MQM 논문에서 제시한 Few-shot 프롬프트를 형식에 맞게 조정"""
    prompts = load_prompts(prompt_path)
    return prompts['gemba_mqm_prompt'].format(source_text=source_text, translation=translation)

# EAPrompt 프롬프트 생성 함수
def create_ea_prompt(source_text: str, translation: str, prompt_path: str) -> str:
    """EAPrompt 논문의 프롬프트 형식에 Few-shot 예제를 추가하고 형식에 맞게 조정"""
    prompts = load_prompts(prompt_path)
    return prompts['ea_prompt'].format(source_text=source_text, translation=translation)
