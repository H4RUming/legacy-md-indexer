"""
Phase 2 Agentic Router
- Query 분석 및 param 추출
- Gradio 5 input 예외처리 및 JSON 파싱
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
            raise FileNotFoundError("catalog.json 누락")
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _clean_query(self, query: Any) -> str:
        # Gradio dict/list payload 대응
        if isinstance(query, list) and len(query) > 0 and isinstance(query[0], dict):
            return query[0].get('text', str(query))
        if isinstance(query, dict):
            return query.get('text', str(query))
        return str(query)

    def _extract_parameters(self, query: str) -> dict:
        # 최근 2년 등 상대 시간 처리를 위해 현재 연도 주입
        current_year = 2026
        
        prompt = f"""
        질문에서 검색 param 추출. JSON만 출력.
        해당 없으면 빈 배열([]) 또는 null.

        [조건]
        - years: 연도 (list of int). 현재 {current_year}년 기준 '최근 2년' 등은 계산해서 배열로 리턴 (예: [2025, 2026])
        - months: 월 (list of int)
        - keyword: 핵심 키워드 (string, 예: "정비", "엘리베이터")

        [질문]
        "{query}"

        [출력 예시]
        {{"years": [2025, 2026], "months": [], "keyword": "엘리베이터"}}
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
            
            # <think> 태그 및 마크다운 노이즈 제거
            clean_text = re.sub(r'<think>.*?</think>', '', res_text, flags=re.DOTALL)
            clean_text = re.sub(r'```json|```', '', clean_text).strip()
            
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                logger.error("JSON parse fail")
                return {"years": [], "months": [], "keyword": None}
                
        except Exception as e:
            logger.error(f"Param extract error: {e}")
            return {"years": [], "months": [], "keyword": None}

    def _filter_catalog(self, params: dict) -> List[str]:
        target_years = params.get("years") or []
        target_months = params.get("months") or []
        keyword = params.get("keyword")

        # 기존 포맷(단일 int) 하위호환용
        if "year" in params and isinstance(params["year"], int):
            target_years.append(params["year"])
        if "month" in params and isinstance(params["month"], int):
            target_months.append(params["month"])

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
        logger.info(f"Filtered docs: {len(target_files)} ea")

        return {
            "query": clean_q,
            "parameters": params,
            "target_files": target_files
        }