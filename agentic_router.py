"""
Phase 2 Agentic Router
- 메타데이터(연도, 월) 기반 Pre-filtering
- 텍스트 키워드 검색은 Phase 3(BM25)로 위임
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
        self.model_id = config['DEFAULT'].get('MODEL_ID', 'qwen2.5:14b')
        self.api_url = config['DEFAULT'].get('OLLAMA_URL', 'http://localhost:11434/api/generate')

    def _load_catalog(self) -> dict:
        if not self.catalog_path.exists():
            raise FileNotFoundError("catalog.json missing")
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _clean_query(self, query: Any) -> str:
        # payload parsing
        if isinstance(query, list) and len(query) > 0 and isinstance(query[0], dict):
            return query[0].get('text', str(query))
        if isinstance(query, dict):
            return query.get('text', str(query))
        return str(query)

    def _extract_parameters(self, query: str) -> dict:
        current_year = 2026
        
        # keyword 제거, search_query(BM25용) 추가
        prompt = f"""
        질문 분석 후 JSON 응답.
        
        [조건]
        - years: 연도 (list of int). 현재 {current_year}년 기준. (예: 최근 2년 -> [2025, 2026])
        - months: 월 (list of int). 조건 없으면 []
        - search_query: 본문 검색용 질의어 (string)
        
        [질문]
        "{query}"
        
        [출력]
        {{"years": [2025, 2026], "months": [], "search_query": "엘리베이터 수리 내역"}}
        """

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }

        try:
            res = requests.post(self.api_url, json=payload, timeout=60)
            res.raise_for_status()
            res_text = res.json().get("response", "")
            
            clean_text = re.sub(r'<think>.*?</think>', '', res_text, flags=re.DOTALL).strip()
            clean_text = re.sub(r'```json|```', '', clean_text).strip()
            
            try:
                return json.loads(clean_text)
            except json.JSONDecodeError:
                match = re.search(r'\{[\s\S]*\}', clean_text)
                if match:
                    return json.loads(match.group(0))
                else:
                    logger.error(f"JSON parse err. raw: {res_text}")
                    return {"years": [], "months": [], "search_query": query}
                
        except Exception as e:
            logger.error(f"Param extract err: {e}")
            return {"years": [], "months": [], "search_query": query}

    def _filter_catalog(self, params: dict) -> List[str]:
        target_years = params.get("years") or []
        target_months = params.get("months") or []
        
        filtered_files = []

        for file_path, meta in self.catalog.items():
            if meta.get("status") != "COMPLETED":
                continue

            year = meta.get("year")
            month = meta.get("month")

            # 시간 조건만 필터링
            if target_years and year not in target_years:
                continue
            if target_months and month not in target_months:
                continue

            filtered_files.append(file_path)

        return filtered_files

    def route_query(self, raw_query: Any) -> Dict[str, Any]:
        clean_q = self._clean_query(raw_query)
        logger.info(f"Query: {clean_q}")
        
        params = self._extract_parameters(clean_q)
        logger.info(f"Extracted params: {params}")

        target_files = self._filter_catalog(params)
        
        # OOM 방어
        MAX_DOCS = 500 
        if len(target_files) > MAX_DOCS:
            logger.warning(f"Docs over limit ({len(target_files)} ea). Truncating to {MAX_DOCS}")
            target_files = target_files[:MAX_DOCS]
            
        logger.info(f"Filtered docs: {len(target_files)} ea")

        return {
            "query": clean_q,
            "parameters": params,
            "target_files": target_files,
            "search_query": params.get("search_query", clean_q)
        }