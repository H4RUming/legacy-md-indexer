"""
Project: Data RAG System
Module: Phase 2 Agentic Router (Pre-filtering)
Description: 
    - 사용자 자연어 질의 분석 및 검색 파라미터 추출 (Function Calling 역할)
    - file_catalog.json 기반 타겟 문서 Pre-filtering
"""

import json
import logging
import requests
import configparser
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
        self.model_id = config['DEFAULT'].get('MODEL_ID', 'nemotron-cascade-2')
        self.api_url = config['DEFAULT'].get('OLLAMA_URL', 'http://localhost:11434/api/generate')

    def _load_catalog(self) -> dict:
        if not self.catalog_path.exists():
            raise FileNotFoundError("catalog 파일이 없습니다. Phase 1을 먼저 실행하십시오.")
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_parameters(self, query: str) -> dict:
        """LLM을 이용해 사용자 질의에서 검색 파라미터 추출"""
        prompt = f"""
        다음 사용자의 질문을 분석하여 검색에 필요한 파라미터를 추출하라.
        조건에 해당하지 않는 값은 null로 처리한다.
        반드시 JSON 포맷으로만 응답하라.

        [추출 조건]
        - year: 연도 (int)
        - month: 월 (int)
        - keyword: 문서 종류나 핵심 키워드 (string, 예: "정비", "주간업무", "STS")

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
            return json.loads(res_text)
        except Exception as e:
            logger.error(f"파라미터 추출 실패: {e}")
            return {"year": None, "month": None, "keyword": None}

    def _filter_catalog(self, params: dict) -> List[str]:
        """추출된 파라미터로 카탈로그 필터링"""
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

            # 필터링 조건 (파라미터가 존재할 경우에만 일치 여부 검사)
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

    def route_query(self, query: str) -> Dict[str, Any]:
        """질의 분석 및 타겟 파일 리스트 반환의 메인 파이프라인"""
        logger.info(f"Query: {query}")
        
        # 1. 파라미터 추출
        params = self._extract_parameters(query)
        logger.info(f"Extracted Params: {params}")

        # 2. Pre-filtering 적용
        target_files = self._filter_catalog(params)
        logger.info(f"Filtered Documents: {len(target_files)} ea")

        return {
            "query": query,
            "parameters": params,
            "target_files": target_files
        }

if __name__ == "__main__":
    router = AgenticRouter(catalog_path="./file_catalog.json")
    
    # 테스트 쿼리
    test_query = "2024년 7월 STS 정비 내역 찾아줘"
    result = router.route_query(test_query)
    
    print("\n[Target Files]")
    for f in result["target_files"]:
        print(f"- {f}")