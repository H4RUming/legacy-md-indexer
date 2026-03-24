"""
Project: Data RAG System
Module: MD Catalog Builder (Track 1 & Sanitizer)
Description: 
    - processed_md 파일 대상 정제 및 메타데이터 1차 추출
    - Crash-safe JSON Dump (단일 파일 처리 단위)
    - Track 1 실패 시 PENDING_LLM 상태로 마킹하여 Track 2 큐로 라우팅
"""

import os
import json
import re
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

# 로거 셋업
logger = logging.getLogger("CATALOG_BUILDER")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('[%(levelname)s] %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

def sanitize_md_for_rag(text: str) -> str:
    """RAG 청킹 최적화를 위한 노이즈 정제 (표, 헤더 마크다운 구조 보존)"""
    # HTML 주석 등 미디어 태그 제거
    text = re.sub(r'', '', text)
    # 특수문자 노이즈 제거 (마크다운 제어문자 보존)
    text = re.sub(r'[^\w\s\n\.,!\?\-\|#\(\)\[\]<>\'":;/%&~+*=]', '', text)
    # 연속 공백 및 탭 정규화
    text = re.sub(r'[ \t]+', ' ', text)
    # 3연속 이상 줄바꿈 압축
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

class RegexFastTrack:
    """Track 1: 정규식 기반 고속 분류기"""
    def __init__(self):
        self.doc_type_patterns = [
            r'(주간업무|업무보고|일일보고)',
            r'(회의록|의사록)',
            r'(견적서|발주서|품의서)',
            r'(검수|정기검사)',
            r'(계획서|기획안)'
        ]
        self.year_pattern = re.compile(r'(20\d{2})(?:년|\.|-|_|$)')
        self.month_pattern = re.compile(r'(?:^|[^\d])(1[0-2]|[1-9])(?:월|\.|-|_|$)')
        self.content_date_pattern = re.compile(r'작성일\s*[:]\s*(20\d{2})[\.\-년\s]+([0-1]?[0-9])[\.\-월\s]')

    def process(self, file_path: Path, content_head: str) -> Dict[str, Optional[str]]:
        year, month = self._extract_date(file_path, content_head)
        doc_type = self._extract_doc_type(file_path)
        
        return {
            "doc_type": doc_type,
            "year": year,
            "month": month,
            "is_complete": bool(year and month and doc_type)
        }

    def _extract_date(self, file_path: Path, content_head: str) -> Tuple[Optional[int], Optional[int]]:
        year, month = None, None
        
        # 본문 상단 헤더 데이터 최우선 파싱
        if content_head:
            match = self.content_date_pattern.search(content_head)
            if match:
                return int(match.group(1)), int(match.group(2))

        # Path 역순 탐색 파싱
        for part in file_path.parts[::-1]:
            clean_part = part.replace(' ', '')
            if not year:
                y_match = self.year_pattern.search(clean_part)
                if y_match: year = int(y_match.group(1))
            
            if not month:
                m_match = self.month_pattern.search(clean_part)
                if m_match: month = int(m_match.group(1))
            
            if year and month:
                break
                
        return year, month

    def _extract_doc_type(self, file_path: Path) -> Optional[str]:
        full_path_str = str(file_path).replace(' ', '')
        for pattern in self.doc_type_patterns:
            match = re.search(pattern, full_path_str)
            if match: return match.group(1)
        return None

class CatalogBuilder:
    def __init__(self, target_dir: str, output_json: str):
        self.target_dir = Path(target_dir)
        self.output_json = Path(output_json)
        self.fast_track = RegexFastTrack()
        self.catalog = self._load_existing_catalog()

    def _load_existing_catalog(self) -> dict:
        """Crash-safe 지원을 위한 카탈로그 로드"""
        if self.output_json.exists():
            try:
                with open(self.output_json, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"JSON 손상. 백업본 생성 필요: {self.output_json.name}")
        return {}

    def _dump_catalog(self):
        """단일 파일 처리 직후 상태 기록"""
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(self.catalog, f, ensure_ascii=False, indent=2)

    def run_pipeline(self):
        if not self.target_dir.exists():
            logger.error(f"타겟 디렉토리 없음: {self.target_dir}")
            return

        md_files = list(self.target_dir.rglob('*.md'))
        logger.info(f"파싱 대상: Total {len(md_files)}ea")

        start_t = time.time()
        track1_hit = 0
        track2_route = 0
        skip_cnt = 0

        for fpath in md_files:
            file_key = str(fpath.relative_to(self.target_dir))
            
            # 이어하기 방어 (성공건 및 LLM 대기건 스킵)
            if file_key in self.catalog:
                status = self.catalog[file_key].get("status")
                if status in ["COMPLETED", "PENDING_LLM"]:
                    skip_cnt += 1
                    continue

            # 파일 Read & Sanitize
            try:
                raw_text = fpath.read_text(encoding='utf-8')
                clean_text = sanitize_md_for_rag(raw_text)
            except Exception as e:
                logger.error(f"파일 Read 에러 [{fpath.name}]: {e}")
                continue

            # 본문 헤더 500자 슬라이싱 (Track 1 리소스 절약)
            content_head = clean_text[:500]
            
            # Track 1 라우팅
            meta = self.fast_track.process(fpath, content_head)
            
            if meta.pop("is_complete"):
                self.catalog[file_key] = {
                    "status": "COMPLETED",
                    "doc_type": meta["doc_type"],
                    "year": meta["year"],
                    "month": meta["month"],
                    "source": "Track1"
                }
                track1_hit += 1
            else:
                # Track 1 실패, Track 2 (LLM Fallback) 대기 상태로 마킹
                self.catalog[file_key] = {
                    "status": "PENDING_LLM",
                    "partial_meta": meta,
                    "source": "Track2_Queue"
                }
                track2_route += 1

            self._dump_catalog()

        elapsed = time.time() - start_t
        logger.info(f"Track 1 파이프라인 종료. (Elapsed: {elapsed:.2f}s)")
        logger.info(f"Result - Track1 Hit: {track1_hit}, Track2 Queue: {track2_route}, Skips: {skip_cnt}")

import requests
import configparser

logger = logging.getLogger("TRACK2_OLLAMA")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s', '%H:%M:%S'))
logger.addHandler(ch)

class OllamaFallbackRouter:
    """Track 2: PENDING_LLM 대상 로컬 Ollama Zero-shot 분류기"""
    def __init__(self, catalog_path: str, target_dir: str):
        self.catalog_path = Path(catalog_path)
        self.target_dir = Path(target_dir)
        self.catalog = self._load_catalog()
        
        # 설정 로드
        config = configparser.ConfigParser()
        config.read('setting.conf')
        self.model_id = config['DEFAULT'].get('MODEL_ID', 'qwen2.5:14b')
        self.api_url = config['DEFAULT'].get('OLLAMA_URL', 'http://localhost:11434/api/generate')

    def _load_catalog(self) -> dict:
        if self.catalog_path.exists():
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        raise FileNotFoundError("catalog 파일 누락. Track 1 선행 필요")

    def _dump_catalog(self):
        with open(self.catalog_path, 'w', encoding='utf-8') as f:
            json.dump(self.catalog, f, ensure_ascii=False, indent=2)

    def _extract_json(self, text: str) -> dict:
        """CoT 태그 제거 및 JSON 블록 추출"""
        clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.error("JSON decode 에러 발생")
        return {}

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048
            }
        }
        try:
            res = requests.post(self.api_url, json=payload, timeout=120)
            res.raise_for_status()
            return res.json().get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API 호출 실패: {e}")
            return ""

    def run(self):
        pending_keys = [k for k, v in self.catalog.items() if v.get("status") == "PENDING_LLM"]
        if not pending_keys:
            logger.info("PENDING_LLM task 없음. 종료.")
            return

        logger.info(f"Ollama Fallback 시작: Total {len(pending_keys)}ea (Model: {self.model_id})")

        for key in pending_keys:
            start_t = time.time()
            fpath = self.target_dir / key
            
            if not fpath.exists():
                logger.warning(f"Target 파일 누락: {key}")
                continue

            # Context 제한 (Max 1500 chars)
            raw_text = fpath.read_text(encoding='utf-8')
            trunc_text = raw_text[:1500]

            prompt = f"""다음 문서의 내용을 파악하여 doc_type, year, month를 추출하라.
알 수 없는 항목은 null로 처리할 것.
반드시 아래 JSON 포맷으로만 응답하라. 설명은 생략한다.

[포맷]
{{"doc_type": "string", "year": "int", "month": "int"}}

[문서내용]
{trunc_text}"""

            res_text = self._call_ollama(prompt)
            if not res_text:
                self.catalog[key]["status"] = "ERROR"
                self._dump_catalog()
                continue

            parsed_data = self._extract_json(res_text)
            
            if parsed_data:
                self.catalog[key].update({
                    "status": "COMPLETED",
                    "doc_type": parsed_data.get("doc_type"),
                    "year": parsed_data.get("year"),
                    "month": parsed_data.get("month"),
                    "source": "Track2_Ollama"
                })
                elapsed = time.time() - start_t
                logger.info(f"Ollama parsing 성공: {key} ({elapsed:.2f}s)")
            else:
                self.catalog[key]["status"] = "FAILED"
                logger.error(f"데이터 추출 실패: {key}")

            # 파일 단위 상태 저장
            self._dump_catalog()

if __name__ == "__main__":
    target = "./processed_md"
    catalog_out = "./file_catalog.json"
    
    # 1. Regex Fast-track 실행
    builder = CatalogBuilder(target_dir=target, output_json=catalog_out)
    builder.run_pipeline()
    
    # 2. Track 1 종료 후, 실패 건들에 대해 Ollama Fallback 실행
    ollama_router = OllamaFallbackRouter(catalog_path=catalog_out, target_dir=target)
    ollama_router.run()