"""
Module: Web GUI
Description:
    - Gradio 5.x message format 대응
    - 통합된 RAGGenerator(BM25 포함) 파이프라인 연결
    - 실시간 스트리밍 및 인터랙션 로깅
"""

import logging
import gradio as gr
import json
import time
from datetime import datetime
from pathlib import Path

from agentic_router import AgenticRouter
from rag_generator import RAGGenerator

logger = logging.getLogger("RAG_GUI")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s', '%H:%M:%S'))
logger.addHandler(ch)

LOG_FILE_PATH = Path("./interaction_logs.json")

custom_css = """
@font-face {
    font-family: 'NanumGothic';
    src: url('file/NanumGothic.ttf') format('truetype');
}
* { font-family: 'NanumGothic', sans-serif !important; }
.gradio-container { max-width: 1400px !important; }
.source-panel { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 16px; font-size: 13px; max-height: 550px; overflow-y: auto; }
.source-item { margin-bottom: 8px; padding: 10px 12px; background: white; border-left: 4px solid #475569; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); word-break: break-all; color: #334155; }
"""

class IntegratedRAGEngine:
    def __init__(self):
        try:
            # Phase 2 Router
            self.router = AgenticRouter(catalog_path="./file_catalog.json")
            # Phase 3 & 4 Integrated Generator (BM25 + RAG)
            self.generator = RAGGenerator(target_dir="./processed_md")
            self._init_log_file()
            logger.info("RAG Engine init success")
        except Exception as e:
            logger.error(f"Engine init fail: {e}")

    def _init_log_file(self):
        if not LOG_FILE_PATH.exists():
            with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def _log_interaction(self, query: str, params: dict, targets: list, answer: str, duration: float, status: str = "COMPLETED"):
        try:
            with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
                logs = json.load(f)

            logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": status,
                "user_query": query,
                "process_duration_s": round(duration, 3),
                "routing_result": {
                    "parameters": params,
                    "target_files_count": len(targets)
                },
                "rag_result": {
                    "answer": answer,
                    "referenced_files": targets
                }
            })

            with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            
            logger.info("Interaction logged")
        except Exception as e:
            logger.error(f"Log write fail: {e}")

    def route_query(self, query: str):
        """메타데이터 기반 1차 필터링"""
        start_t = time.time()
        route_res = self.router.route_query(query)
        return route_res, time.time() - start_t


def build_gradio_ui():
    engine = IntegratedRAGEngine()
    
    example_queries = [
        ["2024년 이동장비 주요 정비 내역 요약"],
        ["최근 2년간 엘리베이터 수리 내역"],
        ["어제 진행된 정비 로그에서 특이사항 정리해 줘"],
        ["교체시기가 오래된 부품들의 리스트를 알려줘"]
    ]

    with gr.Blocks(css=None) as demo:
        gr.HTML(f"""
        <style>{custom_css}</style>
        <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 24px 32px; border-radius: 8px; margin-bottom: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <h1 style="margin: 0; color: #ffffff !important; font-weight: 700; font-size: 26px; letter-spacing: -0.5px; border-bottom: none;">Legacy Document Intelligence Engine</h1>
            <p style="margin: 8px 0 0 0; color: #cbd5e1 !important; font-size: 14px;">BM25 + MD RAG 통합 추론 모드</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(height=600, show_label=False)
                with gr.Row():
                    msg_input = gr.Textbox(show_label=False, placeholder="질의 입력 (Enter)", container=False, scale=7)
                    submit_btn = gr.Button("실행", variant="primary", scale=1)
                    stop_btn = gr.Button("중지", variant="stop", scale=1, interactive=True)
                
                gr.Examples(examples=example_queries, inputs=msg_input, label="예시 질문")
                gr.ClearButton([msg_input, chatbot], value="세션 초기화", size="sm")

            with gr.Column(scale=3):
                gr.Markdown("### 아키텍처 정보")
                with gr.Group():
                    gr.Textbox(value="GPT-OSS:120b (Ollama)", label="Model", interactive=False)
                    gr.Textbox(value="Agentic Router + Integrated BM25", label="Pipeline", interactive=False)
                
                gr.Markdown("### 추출 파라미터")
                params_display = gr.JSON(label="Parameters", value={})
                
                gr.Markdown("### 참조 문서 리스트")
                source_display = gr.HTML(value="<div class='source-panel' style='color: #64748b; text-align: center; padding-top: 30px;'>질의 실행 시 결과가 표출됩니다.</div>")

        def user_interaction(user_message, history):
            history = history or []
            history.append({"role": "user", "content": user_message})
            return "", history

        def bot_interaction_route(history):
            if not history or history[-1]["role"] != "user": 
                return history, "", {}, 0, ""
            
            query = history[-1]["content"]
            route_res, route_duration = engine.route_query(query)
            
            history.append({
                "role": "assistant", 
                "content": "문서 검색 및 답변 생성 중..."
            })
            return history, "", route_res, route_duration, query

        def bot_interaction_generate(history, route_res, route_duration, query):
            if not history or history[-1]["role"] != "assistant": 
                yield history, "", {}
                return
            
            start_gen_t = time.time()
            full_answer = ""
            # 라우터에서 추출한 본문 검색 쿼리 사용
            search_q = route_res.get("search_query", query)
            
            try:
                # generate_stream 내에서 BM25(Kiwi) 연산 후 RAG 수행
                for rag_chunk in engine.generator.generate_stream(query, route_res["target_files"], search_query=search_q):
                    full_answer = rag_chunk["answer"]
                    sources = rag_chunk["references"] # BM25를 거친 최종 Top-K
                    
                    # Source HTML 갱신
                    source_html = "<div class='source-panel'>"
                    if sources:
                        for idx, src in enumerate(sources, 1):
                            fpath = src.get("file_path", src) if isinstance(src, dict) else src
                            score = f" (Score: {src.get('score', 0):.4f})" if isinstance(src, dict) and "score" in src else ""
                            source_html += f"<div> class='source-item'><b>[{idx}]</b> {fpath}{score}</div>"
                    else: 
                        source_html += "<div style='color: #64748b; text-align: center; padding: 20px;'>참조 문서 없음</div>"
                    source_html += "</div>"
                    
                    history[-1]["content"] = full_answer
                    yield history, source_html, route_res.get("parameters", {})
                
                total_duration = route_duration + (time.time() - start_gen_t)
                engine._log_interaction(query, route_res["parameters"], sources, full_answer, total_duration, "COMPLETED")

            except Exception as e:
                logger.warning(f"Inference stopped or error: {e}")
                history[-1]["content"] = full_answer + "\n\n[출력이 중단되었습니다.]"
                total_duration = route_duration + (time.time() - start_gen_t)
                engine._log_interaction(query, route_res["parameters"], route_res["target_files"], history[-1]["content"], total_duration, "STOPPED")
                yield history, "", route_res.get("parameters", {})

        msg_query_state = gr.State()
        route_res_state = gr.State()
        route_dur_state = gr.State()

        # Chain logic
        submit_event = submit_btn.click(user_interaction, [msg_input, chatbot], [msg_input, chatbot], queue=False).then(
            bot_interaction_route, [chatbot], [chatbot, source_display, route_res_state, route_dur_state, msg_query_state], queue=False).then(
            bot_interaction_generate, [chatbot, route_res_state, route_dur_state, msg_query_state], [chatbot, source_display, params_display], queue=True)

        input_event = msg_input.submit(user_interaction, [msg_input, chatbot], [msg_input, chatbot], queue=False).then(
            bot_interaction_route, [chatbot], [chatbot, source_display, route_res_state, route_dur_state, msg_query_state], queue=False).then(
            bot_interaction_generate, [chatbot, route_res_state, route_dur_state, msg_query_state], [chatbot, source_display, params_display], queue=True)
        
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, input_event])
        
    return demo

if __name__ == "__main__":
    logger.info("Starting Web Server")
    app = build_gradio_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["."])