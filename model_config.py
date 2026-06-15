"""
모델 설정 — HuggingFace Inference API
===========================================
- LLM_MODEL: 원하는 분위기를 받아 지브리 감성 여행지를 추천하는 텍스트 LLM

토큰은 .env 파일의 HUGGINGFACEHUB_API_TOKEN 또는 HF_TOKEN 환경변수에서 읽는다.
HF Space에 배포할 때는 Space의 Settings > Secrets 에서 HF_TOKEN 을 등록한다.
"""

from __future__ import annotations

import os

# 지브리 감성 추천용 텍스트 LLM (한국어 및 JSON 성능이 뛰어난 공개 모델)
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def get_token() -> str:
    """환경변수에서 HF 토큰을 읽는다 (LangChain LLM 공통)."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN(또는 HUGGINGFACEHUB_API_TOKEN) 환경변수가 비어 있습니다.\n"
            "  1) https://huggingface.co/settings/tokens 에서 Read 토큰 발급\n"
            "  2) 로컬: .env 에 HF_TOKEN=hf_xxx 추가\n"
            "  3) HF Space: Settings > Secrets 에 HF_TOKEN 등록"
        )
    return token

