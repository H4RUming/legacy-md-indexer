"""
Phase 2 Agentic Router
- Query 분석 및 param 추출
- JSON 파싱 강화 및 Context 폭발 방어 (Max docs limit)
"""
import json
import logging
import requests
import configparser
import re
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger("ROUTER")
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
        self.model_id = config['DEFAULT'].get('MODEL_ID', 'gpt-oss:120b')
        self.api_url = config['DEFAULT'].get('OLLAMA_URL', 'http://localhost:11434/api/generate')

    def _load_catalog(self) -> dict:
        if not self.catalog_path.exists():
            raise FileNotFoundError("catalog.json 누락")
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _clean_query(self, query: Any) -> str:
        # Gradio payload 예외처리
        if isinstance(query, list) and len(query) > 0 and isinstance(query[0], dict):
            return query[0].get('text', str(query))
        if isinstance(query, dict):
            return query.get('text', str(query))
        return str(query)

    def _extract_parameters(self, query: str) -> dict:
        current_year = 2026
        
        prompt = f"""
        사용자 질의에서 검색 파라미터만 추출하여 JSON으로 응답하라. 부가 설명 금지.
        
        [조건]
        - years: 연도 (list of int). 현재 {current_year}년 기준. '최근 2년'이면 [2025, 2026] 처럼 계산.
        - months: 월 (list of int). 조건 없으면 []
        - keyword: 검색어 (string). 조건 없으면 null
        
        [질문]
        "{query}"
        
        [출력]
        {{"years": [2025, 2026], "months": [], "keyword": "엘리베이터"}}
        """

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            # Qwen 등 일부 모델에서 format="json" 옵션이 응답을 망가뜨릴 수 있어 제거 (프롬프트로 통제)
            "options": {"temperature": 0.0}
        }

        try:
            res = requests.post(self.api_url, json=payload, timeout=60)
            res.raise_for_status()
            res_text = res.json().get("response", "")
            
            # 1차 정제: <think> 태그 및 마크다운 노이즈 제거
            clean_text = re.sub(r'<think>.*?</think>', '', res_text, flags=re.DOTALL).strip()
            clean_text = re.sub(r'```json|```', '', clean_text).strip()
            
            try:
                # 1차 시도: 텍스트 전체 JSON 파싱
                return json.loads(clean_text)
            except json.JSONDecodeError:
                # 2차 시도: 정규식으로 {} 블록만 강제 추출
                match = re.search(r'\{[\s\S]*\}', clean_text)
                if match:
                    return json.loads(match.group(0))
                else:
                    # 파싱 실패 시 원본 응답 로그 출력 (디버깅용)
                    logger.error(f"JSON parse fail. Raw response: {res_text}")
                    return {"years": [], "months": [], "keyword": None}
                
        except Exception as e:
            logger.error(f"Param extract error: {e}")
            return {"years": [], "months": [], "keyword": None}

    def _filter_catalog(self, params: dict) -> List[str]:
        target_years = params.get("years") or []
        target_months = params.get("months") or []
        keyword = params.get("keyword")

        # 필터 조건이 아예 없는 경우 (파싱 실패 등), 풀스캔 방지
        if not target_years and not target_months and not keyword:
            logger.warning("검색 조건 없음. 풀스캔 방지를 위해 빈 리스트 반환")
            return []

        filtered_files = []

        for file_path, meta in self.catalog.items():
            if meta.get("status") != "COMPLETED":
                continue

            doc_type = meta.get("doc_type", "")
            year = meta.get("year")
            month = meta.get("month")

            if target_years and year not in target_years:
                continue
            if target_months and month not in target_months:
                continue
            if keyword and (keyword.lower() not in doc_type.lower() and keyword.lower() not in file_path.lower()):
                continue

            filtered_files.append(file_path)

        return filtered_files

    def route_query(self, raw_query: Any) -> Dict[str, Any]:
        clean_q = self._clean_query(raw_query)
        logger.info(f"Query: {clean_q}")
        
        params = self._extract_parameters(clean_q)
        logger.info(f"Extracted params: {params}")

        target_files = self._filter_catalog(params)
        
        # OOM(Out of Memory) 및 Context limit 방어: 최대 문서 수 제한
        MAX_DOCS = 50 
        if len(target_files) > MAX_DOCS:
            logger.warning(f"검색된 문서가 너무 많습니다. ({len(target_files)} ea). {MAX_DOCS}개로 제한(Truncate)합니다.")
            target_files = target_files[:MAX_DOCS]
            
        logger.info(f"Filtered docs: {len(target_files)} ea")

        return {
            "query": clean_q,
            "parameters": params,
            "target_files": target_files
        }