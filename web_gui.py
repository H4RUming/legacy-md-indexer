"""
Module: Web GUI
Description:
    - Gradio 5.x/6.x message format 대응
    - 통합된 RAGGenerator(BM25 포함) 파이프라인 연결
    - 실시간 스트리밍 및 인터랙션 로깅
    - 다크 엔터프라이즈 테마 + 파이프라인 상태 인디케이터
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

# ── CSS: Dark Enterprise Theme ────────────────────────────────────────────────
custom_css = """
:root {
    --bg-base:      #0f172a;
    --bg-panel:     #1e293b;
    --bg-card:      #263348;
    --border:       #334155;
    --border-light: rgba(51,65,85,0.5);
    --accent:       #38bdf8;
    --accent-glow:  #0ea5e9;
    --stage1:       #f59e0b;
    --stage2:       #10b981;
    --error:        #f87171;
    --text-p:       #f1f5f9;
    --text-s:       #94a3b8;
    --text-m:       #64748b;
}

@font-face {
    font-family: 'NanumGothic';
    src: url('file/NanumGothic.ttf') format('truetype');
}
* { font-family: 'NanumGothic', 'Noto Sans KR', sans-serif !important; }

.gradio-container { max-width: 1440px !important; }

/* ── Header ───────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #0c1629 0%, #0f2a4a 50%, #1a2744 100%);
    border: 1px solid #1e40af;
    border-radius: 10px;
    padding: 22px 32px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(56,189,248,0.08);
}
.main-header h1 {
    margin: 0 0 6px 0;
    color: var(--text-p) !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    letter-spacing: -0.4px;
    border-bottom: none !important;
}
.main-header .subtitle {
    color: var(--text-s) !important;
    font-size: 13px;
    line-height: 1.6;
    margin: 0 0 8px 0;
}
.main-header .badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: var(--accent) !important;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── Pipeline Status Indicator ───────────────────────── */
.status-panel {
    border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 10px;
    min-height: 42px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    color: var(--text-s);
    background: var(--bg-panel);
}
.status-panel.idle   { border-left-color: var(--border); color: var(--text-m); }
.status-panel.stage1 { border-left-color: var(--stage1); }
.status-panel.stage2 { border-left-color: var(--stage2); }
.status-panel.done   { border-left-color: var(--accent); color: var(--text-s); }
.status-panel.error  { border-left-color: var(--error); color: var(--error) !important; }

.spinner {
    width: 13px; height: 13px;
    border: 2px solid rgba(255,255,255,0.1);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
    flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Right Panel ─────────────────────────────────────── */
.right-panel-wrap {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
}

.panel-label {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    color: var(--text-m) !important;
    margin: 14px 0 8px 0 !important;
    padding-bottom: 5px !important;
    border-bottom: 1px solid var(--border) !important;
}
.panel-label:first-child { margin-top: 0 !important; }

.arch-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid var(--border-light);
    font-size: 12px;
}
.arch-row:last-child { border-bottom: none; }
.arch-label { color: var(--text-m); }
.arch-value { color: var(--accent); font-weight: 600; font-size: 11px; }

/* ── Timing Cards ────────────────────────────────────── */
.timing-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin-top: 4px;
}
.timing-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 6px;
    text-align: center;
}
.timing-val {
    font-size: 17px;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.timing-lbl {
    font-size: 10px;
    color: var(--text-m);
    margin-top: 4px;
    letter-spacing: 0.3px;
}

/* ── Param Rows ─────────────────────────────────────── */
.param-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 5px 0;
    border-bottom: 1px solid var(--border-light);
    font-size: 12px;
    gap: 8px;
}
.param-row:last-child { border-bottom: none; }
.param-label { color: var(--text-m); flex-shrink: 0; }
.param-value { color: var(--text-s); font-size: 11px; text-align: right; word-break: break-all; }

/* ── Source Documents ────────────────────────────────── */
.source-panel {
    max-height: 320px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}
.source-panel::-webkit-scrollbar { width: 3px; }
.source-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.source-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-glow);
    border-radius: 5px;
    padding: 8px 10px;
    margin-bottom: 5px;
    font-size: 11px;
    color: var(--text-s);
    word-break: break-all;
    line-height: 1.5;
}
.source-item:last-child { margin-bottom: 0; }
.source-num { color: var(--accent); font-weight: 700; margin-right: 4px; }
.source-score {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    color: var(--accent);
    font-size: 10px;
    padding: 1px 5px;
    border-radius: 3px;
    margin-left: 4px;
    vertical-align: middle;
}
.source-empty {
    color: var(--text-m);
    text-align: center;
    padding: 20px 0;
    font-size: 12px;
}
"""

# ── Status HTML Helpers ───────────────────────────────────────────────────────
def _status_idle() -> str:
    return "<div class='status-panel idle'>대기 중</div>"

def _status_stage1() -> str:
    return """<div class='status-panel stage1'>
        <div class='spinner'></div>
        <span>Stage 1 &mdash; 아젠틱 라우터 실행 중 (메타데이터 필터링)</span>
    </div>"""

def _status_stage1_done(duration: float, count: int) -> str:
    return f"""<div class='status-panel stage1' style='border-left-color:#10b981;'>
        ✓ Stage 1 완료 &mdash; {count}개 문서 선별 ({duration:.2f}s)
    </div>"""

def _status_stage2() -> str:
    return """<div class='status-panel stage2'>
        <div class='spinner' style='border-top-color:#10b981;'></div>
        <span>Stage 2 &mdash; BM25 검색 및 LLM 답변 생성 중</span>
    </div>"""

def _status_complete(route_dur: float, gen_dur: float) -> str:
    total = route_dur + gen_dur
    return f"""<div class='status-panel done'>
        ✓ 완료 &mdash; 라우팅 {route_dur:.2f}s &nbsp;/&nbsp; 생성 {gen_dur:.2f}s &nbsp;/&nbsp; 총 {total:.2f}s
    </div>"""

def _status_error(msg: str) -> str:
    return f"<div class='status-panel error'>오류: {msg}</div>"


# ── Source HTML Helper ────────────────────────────────────────────────────────
def _build_source_html(sources: list) -> str:
    if not sources:
        return "<div class='source-panel'><div class='source-empty'>참조 문서 없음</div></div>"
    html = "<div class='source-panel'>"
    for idx, src in enumerate(sources, 1):
        fpath = src.get("file_path", src) if isinstance(src, dict) else str(src)
        score_val = src.get("score") if isinstance(src, dict) else None
        score_html = f"<span class='source-score'>{score_val:.4f}</span>" if score_val is not None else ""
        html += f"<div class='source-item'><span class='source-num'>[{idx}]</span>{fpath}{score_html}</div>"
    html += "</div>"
    return html


# ── Stats/Params HTML Helpers ─────────────────────────────────────────────────
def _stats_empty() -> str:
    return "<div class='panel-label'>파이프라인 통계</div><div class='source-empty'>질의 실행 후 표출</div>"

def _stats_html(route_dur: float, gen_dur: float, params: dict, doc_count: int) -> str:
    total = route_dur + gen_dur
    years_str  = ", ".join(str(y) for y in params.get("years",  [])) or "전체"
    months_str = ", ".join(str(m) for m in params.get("months", [])) or "전체"
    search_q   = params.get("search_query", "-") or "-"
    return f"""
<div class='panel-label'>파이프라인 통계</div>
<div class='timing-grid'>
  <div class='timing-card'><div class='timing-val'>{route_dur:.1f}s</div><div class='timing-lbl'>라우팅</div></div>
  <div class='timing-card'><div class='timing-val'>{gen_dur:.1f}s</div><div class='timing-lbl'>생성</div></div>
  <div class='timing-card'><div class='timing-val'>{total:.1f}s</div><div class='timing-lbl'>총 소요</div></div>
</div>
<div class='panel-label' style='margin-top:14px;'>추출 파라미터</div>
<div class='param-row'><span class='param-label'>연도</span><span class='param-value'>{years_str}</span></div>
<div class='param-row'><span class='param-label'>월</span><span class='param-value'>{months_str}</span></div>
<div class='param-row'><span class='param-label'>검색어</span><span class='param-value'>{search_q}</span></div>
<div class='param-row'><span class='param-label'>필터 문서</span><span class='param-value'>{doc_count}개</span></div>
"""


# ── Integrated RAG Engine ─────────────────────────────────────────────────────
class IntegratedRAGEngine:
    def __init__(self):
        self.ready = False
        self.init_error = ""
        catalog_path = Path("./file_catalog.json")

        if not catalog_path.exists():
            self.init_error = (
                "file_catalog.json 파일이 없습니다. "
                "먼저 md_catalog_builder.py를 실행하여 카탈로그를 생성하세요."
            )
            logger.error(self.init_error)
            return

        try:
            self.router = AgenticRouter(catalog_path=str(catalog_path))
            self.generator = RAGGenerator(target_dir="./processed_md")
            self._init_log_file()
            logger.info("RAG Engine init success")
            self.ready = True
        except Exception as e:
            self.init_error = f"엔진 초기화 실패: {e}"
            logger.error(self.init_error)

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
        start_t = time.time()
        route_res = self.router.route_query(query)
        return route_res, time.time() - start_t


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_gradio_ui():
    engine = IntegratedRAGEngine()

    example_queries = [
        ["2024년 이동장비 주요 정비 내역 요약"],
        ["최근 2년간 엘리베이터 수리 내역"],
        ["교체시기가 오래된 부품들의 리스트를 알려줘"],
        ["교체가 잦은 부품들의 리스트를 보여주세요"],
        ["24년 예방정비 실적"],
        ["트위스트락 교체 이력을 알려줘"],
        ["COSCO Shipping 업무협의 내용"],
    ]

    # 우측 패널 — 정적 아키텍처 정보
    arch_info_html = """
<div class='right-panel-wrap'>
  <div class='panel-label'>시스템 구성</div>
  <div class='arch-row'><span class='arch-label'>LLM</span><span class='arch-value'>GPT-OSS:120b (Ollama)</span></div>
  <div class='arch-row'><span class='arch-label'>Router</span><span class='arch-value'>Agentic (Metadata)</span></div>
  <div class='arch-row'><span class='arch-label'>Retrieval</span><span class='arch-value'>BM25 + Kiwi</span></div>
  <div class='arch-row'><span class='arch-label'>Context</span><span class='arch-value'>128k tokens</span></div>
</div>
"""

    with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
        # Header
        gr.HTML(f"""
        <div class='main-header'>
            <h1>Legacy Document Intelligence Engine</h1>
            <p class='subtitle'>
                현재 프로토타입 버전으로, 불안정하거나 잘못된 정보를 제공할 수 있습니다.<br>
                참조 문서의 양이 방대한 경우 추론에 시간이 소요될 수 있습니다.
            </p>
            <span class='badge'>Prototype v0.1</span>
        </div>
        """)

        with gr.Row():
            # ── 좌측: 채팅 ──────────────────────────────────────────────
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(height=580, show_label=False)
                with gr.Row():
                    msg_input = gr.Textbox(
                        show_label=False,
                        placeholder="질의 입력 (Enter로 실행)",
                        container=False,
                        scale=7,
                    )
                    submit_btn = gr.Button("실행", variant="primary", scale=1)
                    stop_btn   = gr.Button("중지", variant="stop",    scale=1, interactive=True)

                gr.Examples(examples=example_queries, inputs=msg_input, label="예시 질문")
                gr.ClearButton([msg_input, chatbot], value="세션 초기화", size="sm")

            # ── 우측: 정보 패널 ─────────────────────────────────────────
            with gr.Column(scale=3):
                gr.HTML(value=arch_info_html)

                status_display = gr.HTML(
                    value=_status_idle(),
                )

                stats_display = gr.HTML(
                    value=_stats_empty(),
                )

                gr.HTML("<div class='panel-label' style='margin-top:14px;'>참조 문서 리스트</div>")
                source_display = gr.HTML(
                    value="<div class='source-panel'><div class='source-empty'>질의 실행 시 결과가 표출됩니다.</div></div>",
                )

        # ── State ────────────────────────────────────────────────────────
        msg_query_state  = gr.State()
        route_res_state  = gr.State()
        route_dur_state  = gr.State()

        # ── Event Functions ───────────────────────────────────────────────
        def user_interaction(user_message, history):
            history = history or []
            history.append({"role": "user", "content": user_message})
            return "", history

        def bot_interaction_route(history):
            """제너레이터: 라우팅 시작 전 스피너 즉시 표시"""
            if not history or history[-1]["role"] != "user":
                yield history, _status_idle(), {}, 0, ""
                return

            # 엔진 초기화 실패 가드
            if not engine.ready:
                history.append({"role": "assistant", "content": f"[시스템 오류] {engine.init_error}"})
                yield history, _status_error(engine.init_error), {}, 0, ""
                return

            query = history[-1]["content"]
            history.append({"role": "assistant", "content": "문서 검색 및 답변 생성 중..."})

            # Stage 1 스피너 즉시 표시
            yield history, _status_stage1(), {}, 0, query

            # 블로킹 라우팅 실행
            route_res, route_duration = engine.route_query(query)
            target_count = len(route_res.get("target_files", []))
            yield history, _status_stage1_done(route_duration, target_count), route_res, route_duration, query

        def bot_interaction_generate(history, route_res, route_duration, query):
            if not history or history[-1]["role"] != "assistant":
                yield history, _build_source_html([]), _stats_empty(), _status_idle()
                return

            start_gen_t = time.time()
            full_answer = ""
            sources     = []
            search_q    = route_res.get("search_query", query)

            # Stage 2 스피너 즉시 표시
            yield history, _build_source_html([]), _stats_empty(), _status_stage2()

            try:
                for rag_chunk in engine.generator.generate_stream(
                    query, route_res["target_files"], search_query=search_q
                ):
                    full_answer = rag_chunk["answer"]
                    sources     = rag_chunk["references"]

                    history[-1]["content"] = full_answer
                    yield history, _build_source_html(sources), _stats_empty(), _status_stage2()

                gen_dur = time.time() - start_gen_t
                final_stats  = _stats_html(route_duration, gen_dur, route_res.get("parameters", {}), len(route_res.get("target_files", [])))
                final_status = _status_complete(route_duration, gen_dur)
                yield history, _build_source_html(sources), final_stats, final_status

                engine._log_interaction(
                    query, route_res["parameters"], sources,
                    full_answer, route_duration + gen_dur, "COMPLETED"
                )

            except Exception as e:
                logger.warning(f"Inference stopped or error: {e}")
                history[-1]["content"] = full_answer + "\n\n[출력이 중단되었습니다.]"
                gen_dur = time.time() - start_gen_t
                engine._log_interaction(
                    query, route_res.get("parameters", {}), route_res.get("target_files", []),
                    history[-1]["content"], route_duration + gen_dur, "STOPPED"
                )
                yield history, _build_source_html(sources), _stats_empty(), _status_error(str(e))

        # ── Event Chain ───────────────────────────────────────────────────
        submit_event = (
            submit_btn.click(
                user_interaction,
                [msg_input, chatbot],
                [msg_input, chatbot],
                queue=False,
            )
            .then(
                bot_interaction_route,
                [chatbot],
                [chatbot, status_display, route_res_state, route_dur_state, msg_query_state],
                queue=True,
                show_progress="hidden",
            )
            .then(
                bot_interaction_generate,
                [chatbot, route_res_state, route_dur_state, msg_query_state],
                [chatbot, source_display, stats_display, status_display],
                queue=True,
                show_progress="hidden",
            )
        )

        input_event = (
            msg_input.submit(
                user_interaction,
                [msg_input, chatbot],
                [msg_input, chatbot],
                queue=False,
            )
            .then(
                bot_interaction_route,
                [chatbot],
                [chatbot, status_display, route_res_state, route_dur_state, msg_query_state],
                queue=True,
                show_progress="hidden",
            )
            .then(
                bot_interaction_generate,
                [chatbot, route_res_state, route_dur_state, msg_query_state],
                [chatbot, source_display, stats_display, status_display],
                queue=True,
                show_progress="hidden",
            )
        )

        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, input_event])

    return demo


if __name__ == "__main__":
    logger.info("Starting Web Server")
    app = build_gradio_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True, allowed_paths=["."])
