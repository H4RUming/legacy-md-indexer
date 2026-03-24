"""
Project: Data RAG System
Module: Phase 3 RAG Generator
Description: 
    - Pre-filtered 문서 컨텍스트 전체 로드 (1M Token 지원 모델 대응)
    - 답변 내용과 참조 파일(Reference) 명확적 분리 출력
    - 대규모 Context 주입에 따른 Timeout 방어 및 num_ctx 유연화
"""

import logging
import requests
import configparser
from pathlib import Path
from typing import List, Dict, Any

# 로거 셋업
logger = logging.getLogger("RAG_GENERATOR")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s', '%H:%M:%S'))
logger.addHandler(ch)

class RAGGenerator:
    def __init__(self, target_dir: str):
        self.target_dir = Path(target_dir)
        
        config = configparser.ConfigParser()
        config.read('setting.conf')
        self.model_id = config['DEFAULT'].get('MODEL_ID', 'nemotron-cascade-2')
        self.api_url = config['DEFAULT'].get('OLLAMA_URL', 'http://localhost:11434/api/generate')
        # 1M 토큰 대응 Context 및 타임아웃 세팅
        self.num_ctx = int(config['DEFAULT'].get('NUM_CTX', 1048576)) 
        self.req_timeout = int(config['DEFAULT'].get('REQ_TIMEOUT', 600))

    def _load_context(self, target_files: List[str]) -> str:
        """타겟 마크다운 파일 전체 원문을 하나의 Context 스트링으로 병합"""
        context_blocks = []
        for file_path in target_files:
            fpath = self.target_dir / file_path
            if fpath.exists():
                # 1M 토큰 지원: Truncation 없이 원문 통째로 로드
                text = fpath.read_text(encoding='utf-8')
                context_blocks.append(f"--- [Document: {file_path}] ---\n{text}")
            else:
                logger.warning(f"File missing: {file_path}")
                
        return "\n\n".join(context_blocks)

    def generate(self, query: str, target_files: List[str]) -> Dict[str, Any]:
        """최종 답변 생성 및 출처 분리 반환"""
        if not target_files:
            logger.warning("Target file 리스트 Empty")
            return {
                "answer": "카탈로그 조건에 부합하는 문서가 없어 답변할 수 없습니다.",
                "references": []
            }

        logger.info(f"Load context from {len(target_files)} files...")
        context = self._load_context(target_files)
        
        prompt = f"""다음 제공된 [Context] 문서들만을 참고하여 [Query]에 대한 답변을 작성하라.
Context에 없는 내용은 절대 지어내지 말고, "해당 내용은 문서에서 확인할 수 없습니다"라고 답하라.
설명은 간결하고 핵심만 작성하라.

[Context]
{context}

[Query]
{query}"""

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": self.num_ctx
            }
        }

        logger.info(f"Ollama inference req... (Timeout: {self.req_timeout}s)")
        try:
            res = requests.post(self.api_url, json=payload, timeout=self.req_timeout)
            res.raise_for_status()
            answer = res.json().get("response", "").strip()
            
            return {
                "answer": answer,
                "references": target_files
            }
            
        except requests.exceptions.Timeout:
            logger.error("Inference Timeout: Context 연산에 시간이 너무 오래 걸림.")
            return {
                "answer": "문서 용량이 너무 커서 답변 생성 시간을 초과했습니다.",
                "references": target_files
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Inference error: {e}")
            return {
                "answer": f"답변 생성 중 에러 발생: {e}",
                "references": target_files
            }

if __name__ == "__main__":
    from agentic_router import AgenticRouter
    
    router = AgenticRouter(catalog_path="./file_catalog.json")
    query = "2024년 7월 STS 정비 내역 요약해줘"
    route_result = router.route_query(query)
    
    generator = RAGGenerator(target_dir="./processed_md")
    rag_result = generator.generate(
        query=route_result["query"], 
        target_files=route_result["target_files"]
    )
    
    print("\n" + "="*50)
    print("[AI Answer]")
    print(rag_result["answer"])
    print("-" * 50)
    print("[Referenced Files]")
    for ref in rag_result["references"]:
        print(f"- {ref}")
    print("="*50 + "\n")