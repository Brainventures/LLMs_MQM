from utils.logger_util import setup_logger

import pandas as pd
import logging

logger = logging.getLogger()

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

