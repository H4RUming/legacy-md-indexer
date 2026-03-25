"""
Phase 3 RAG Generator
- Streaming 응답 처리
- Context Window(128k) 대응 및 사전 Truncation
"""
import logging
import requests
import configparser
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Generator, Union

logger = logging.getLogger("RAG_GEN")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s', '%H:%M:%S'))
logger.addHandler(ch)

class RAGGenerator:
    def __init__(self, target_dir: str):
        self.target_dir = Path(target_dir)
        
        config = configparser.ConfigParser()
        config.read('setting.conf')
        self.model_id = config['DEFAULT'].get('MODEL_ID', 'qwen2.5:14b')
        self.api_url = config['DEFAULT'].get('OLLAMA_URL', 'http://localhost:11434/api/generate')
        
        # 128k context 세팅
        self.num_ctx = 131072
        self.req_timeout = 600
        # Context window 방어선 (약 160k chars)
        self.max_char_limit = 160000 

    def _load_context(self, target_files: Union[List[str], List[Dict]]) -> str:
        context_blocks = []
        current_len = 0
        
        # Phase 3(BM25)의 dict 결과물과 단순 str 리스트 모두 호환
        paths = []
        for item in target_files:
            if isinstance(item, dict) and "file_path" in item:
                paths.append(item["file_path"])
            else:
                paths.append(item)
        
        for file_path in paths:
            fpath = self.target_dir / file_path
            if fpath.exists():
                text = fpath.read_text(encoding='utf-8')
                block = f"--- [Doc: {file_path}] ---\n{text}\n\n"
                
                if current_len + len(block) > self.max_char_limit:
                    allowed_len = self.max_char_limit - current_len
                    if allowed_len > 100:
                        context_blocks.append(block[:allowed_len] + "\n...[Truncated]...")
                    logger.warning(f"Context 최대 길이 초과 ({self.max_char_limit} chars). 이후 텍스트 Truncate 처리")
                    break
                
                context_blocks.append(block)
                current_len += len(block)
            else:
                logger.warning(f"참조 문서 누락: {file_path}")
                
        return "".join(context_blocks)

    def generate_stream(self, query: str, target_files: Union[List[str], List[Dict]]) -> Generator[Dict[str, Any], None, None]:
        if not target_files:
            yield {"answer": "조건에 부합하는 문서가 없어 답변할 수 없습니다.", "references": []}
            return

        context = self._load_context(target_files)
        logger.info(f"Context 로드 완료 (length: {len(context)} chars)")
        
        prompt = f"""다음 제공된 [Context] 문서들만 참고해서 [Query]에 대한 답변 작성.
Context에 없는 내용은 지어내지 말고, "해당 내용은 문서에서 확인할 수 없습니다"라고 할 것.
설명은 간결하고 핵심만.

[Context]
{context}

[Query]
{query}"""

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_ctx": self.num_ctx
            }
        }

        logger.info("Ollama stream inference 요청")
        start_t = time.time()
        first_token_received = False

        try:
            with requests.post(self.api_url, json=payload, stream=True, timeout=self.req_timeout) as res:
                res.raise_for_status()
                
                full_answer = ""
                for line in res.iter_lines():
                    if line:
                        if not first_token_received:
                            ttft = time.time() - start_t
                            logger.info(f"TTFT (First Token): {ttft:.2f}s")
                            first_token_received = True

                        chunk = json.loads(line.decode('utf-8'))
                        full_answer += chunk.get("response", "")
                        
                        yield {
                            "answer": full_answer,
                            "references": target_files
                        }
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"API 통신 에러: {e}")
            yield {
                "answer": f"답변 생성 중 에러 발생: {e}",
                "references": target_files
            }