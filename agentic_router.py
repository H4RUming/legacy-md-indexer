"""
Module: Phase 2 Agentic Router (Pre-filtering)
Description: 
    - 사용자 자연어 질의 분석 및 검색 파라미터 추출
    - Gradio 5.x 멀티모달 Input 방어 및 LLM JSON 응답 정규식 추출 추가
"""

import json
import logging
import requests
import configparser
import re
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger("AGENTIC_ROUTER")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.addHandler(ch)

class AgenticRouter:
    def __init__(self, catalog_path: str):
        self.catalog_path = Path(catalog_path)
        self.catalog = self._load_catalog()
        
        config = configparser.ConfigParser()
        config.read('setting.conf')
        self.model_id = config['DEFAULT'].get('MODEL_ID', 'qwen2.5:14b')
        self.api_url = config['DEFAULT'].get('OLLAMA_URL', 'http://localhost:11434/api/generate')

    def _load_catalog(self) -> dict:
        if not self.catalog_path.exists():
            raise FileNotFoundError("catalog 누락. Phase 1 선행 필요")
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _clean_query(self, query: Any) -> str:
        """Gradio 5.x의 dict/list 형태 메시지를 순수 String으로 정제"""
        if isinstance(query, list) and len(query) > 0 and isinstance(query[0], dict):
            return query[0].get('text', str(query))
        if isinstance(query, dict):
            return query.get('text', str(query))
        return str(query)

    def _extract_parameters(self, query: str) -> dict:
        prompt = f"""
        다음 사용자의 질문을 분석하여 검색에 필요한 파라미터를 추출하라.
        조건에 해당하지 않는 값은 null로 처리한다.
        반드시 JSON 포맷으로만 응답하라. 설명은 생략한다.

        [추출 조건]
        - year: 연도 (int)
        - month: 월 (int)
        - keyword: 문서 종류나 핵심 키워드 (string, 예: "정비", "주간업무", "엘리베이터")

        [사용자 질문]
        "{query}"

        [출력 포맷]
        {{"year": 2024, "month": 7, "keyword": "STS"}}
        """

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0}
        }

        try:
            res = requests.post(self.api_url, json=payload, timeout=60)
            res.raise_for_status()
            res_text = res.json().get("response", "")
            
            # 마크다운/사고 과정 태그 등 노이즈 제거 후 순수 JSON 추출
            clean_text = re.sub(r'<think>.*?</think>', '', res_text, flags=re.DOTALL)
            clean_text = re.sub(r'```json|```', '', clean_text).strip()
            
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                logger.error("JSON 포맷 매칭 실패")
                return {"year": None, "month": None, "keyword": None}
                
        except Exception as e:
            logger.error(f"파라미터 추출 실패: {e}")
            return {"year": None, "month": None, "keyword": None}

    def _filter_catalog(self, params: dict) -> List[str]:
        target_year = params.get("year")
        target_month = params.get("month")
        keyword = params.get("keyword")

        filtered_files = []

        for file_path, meta in self.catalog.items():
            if meta.get("status") != "COMPLETED":
                continue

            doc_type = meta.get("doc_type") or ""
            year = meta.get("year")
            month = meta.get("month")

            match = True
            if target_year and year != target_year:
                match = False
            if target_month and month != target_month:
                match = False
            if keyword and keyword.lower() not in doc_type.lower() and keyword.lower() not in file_path.lower():
                match = False

            if match:
                filtered_files.append(file_path)

        return filtered_files

    def route_query(self, raw_query: Any) -> Dict[str, Any]:
        # 입력값 정제
        clean_q = self._clean_query(raw_query)
        logger.info(f"Query: {clean_q}")
        
        params = self._extract_parameters(clean_q)
        logger.info(f"Extracted Params: {params}")

        target_files = self._filter_catalog(params)
        logger.info(f"Filtered Documents: {len(target_files)} ea")

        return {
            "query": clean_q,
            "parameters": params,
            "target_files": target_files
        }