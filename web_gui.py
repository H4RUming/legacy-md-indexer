"""
Module: Web GUI
Description:
    - Gradio 5.x/6.x message format 대응
    - 통합된 RAGGenerator(BM25 포함) 파이프라인 연결
    - 실시간 스트리밍 및 인터랙션 로깅
    - 질문 이력 (등급별 분리) 지원
"""

import os
import hashlib
import logging
import gradio as gr
import json
import time
import uuid
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
USERS_FILE = Path("./users.json")


# ── User Management ──────────────────────────────────────────────────────────
def load_users():
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, rank):
    if not username or not password:
        return "아이디와 비밀번호를 입력하세요."
    users = load_users()
    if username in users:
        return "이미 존재하는 아이디입니다."
    users[username] = {
        "password": hash_password(password),
        "rank": rank
    }
    save_users(users)
    return "회원가입 완료. 로그인 탭에서 진행해주세요."

def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "존재하지 않는 아이디입니다.", ""
    if users[username]["password"] != hash_password(password):
        return False, "비밀번호가 일치하지 않습니다.", ""
    return True, f"{username}님 환영합니다.", users[username]["rank"]


# ── Query History ────────────────────────────────────────────────────────────
def _load_history_html(user_rank: str, limit: int = 20) -> str:
    """로그 파일에서 동일 등급 질문 이력을 HTML로 반환"""
    if not LOG_FILE_PATH.exists():
        return "<div class='history-empty'>아직 질문 이력이 없습니다.</div>"
    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except Exception:
        return "<div class='history-empty'>이력을 불러올 수 없습니다.</div>"

    # 동일 등급만 필터링 후 최신순 정렬
    filtered = [l for l in logs if l.get("user", {}).get("rank") == user_rank]
    filtered = filtered[-limit:][::-1]

    if not filtered:
        return "<div class='history-empty'>아직 질문 이력이 없습니다.</div>"

    html = "<div class='history-list'>"
    for entry in filtered:
        ts = entry.get("timestamp", "")
        user = entry.get("user", {}).get("username", "")
        query = entry.get("query", {}).get("raw_input", "")
        status = entry.get("status", "")
        total = entry.get("total_duration_s", 0)
        doc_cnt = entry.get("stage1_routing", {}).get("filtered_files_count", 0)

        status_cls = "done" if status == "COMPLETED" else "stopped"
        status_ico = "check-ico" if status == "COMPLETED" else "stop-ico"

        # 질문 텍스트가 길면 자르기
        display_q = query if len(query) <= 50 else query[:47] + "..."

        html += f"""<div class='history-item {status_cls}'>
            <div class='history-top'>
                <span class='history-user'>{user}</span>
                <span class='history-time'>{ts}</span>
            </div>
            <div class='history-query'>{display_q}</div>
            <div class='history-meta'>
                <span class='{status_ico}'></span>
                <span>{total:.1f}초 / {doc_cnt}건 검색</span>
            </div>
        </div>"""
    html += "</div>"
    return html


# ── CSS ──────────────────────────────────────────────────────────────────────
custom_css = """
:root, [data-theme="dark"] {
    --bg-base:      #0a0f1e;
    --bg-panel:     #111827;
    --bg-card:      #1a2236;
    --bg-card-h:    #1e2a42;
    --border:       #1f2937;
    --border-light: rgba(31,41,55,0.6);
    --accent:       #3b82f6;
    --accent-dim:   rgba(96,165,250,0.15);
    --green:        #34d399;
    --green-dim:    rgba(52,211,153,0.12);
    --amber:        #fbbf24;
    --amber-dim:    rgba(251,191,36,0.12);
    --red:          #f87171;
    --red-dim:      rgba(248,113,113,0.10);
    --text-p:       #f1f5f9;
    --text-s:       #94a3b8;
    --text-m:       #64748b;
    --radius:       10px;
    --header-bg:    linear-gradient(135deg, #070c1a 0%, #0d1a33 50%, #111f3a 100%);
    --theme-icon:   "🌙";
}

[data-theme="light"] {
    --bg-base:      #f0f2f5;
    --bg-panel:     #ffffff;
    --bg-card:      #f8f9fb;
    --bg-card-h:    #eef1f6;
    --border:       #d1d5db;
    --border-light: rgba(209,213,219,0.6);
    --accent:       #2563eb;
    --accent-dim:   rgba(37,99,235,0.10);
    --green:        #059669;
    --green-dim:    rgba(5,150,105,0.10);
    --amber:        #d97706;
    --amber-dim:    rgba(217,119,6,0.10);
    --red:          #dc2626;
    --red-dim:      rgba(220,38,38,0.08);
    --text-p:       #111827;
    --text-s:       #4b5563;
    --text-m:       #6b7280;
    --header-bg:    linear-gradient(135deg, #e8ecf4 0%, #dfe6f0 50%, #edf0f7 100%);
    --theme-icon:   "☀️";
}

/* ── Theme Toggle ───────────────────────────────────── */
.theme-toggle {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-s);
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
    padding: 0;
    line-height: 1;
}
.theme-toggle:hover {
    background: var(--accent-dim);
    border-color: var(--accent);
    color: var(--accent);
}

/* ── Force light-theme overrides for Gradio internals ── */
[data-theme="light"] .gradio-container {
    background: var(--bg-base) !important;
}
[data-theme="light"] input,
[data-theme="light"] textarea,
[data-theme="light"] select {
    background: var(--bg-panel) !important;
    color: var(--text-p) !important;
    border-color: var(--border) !important;
}
[data-theme="light"] .chatbot {
    background: var(--bg-panel) !important;
}
[data-theme="light"] button.primary {
    background: var(--accent) !important;
}
[data-theme="light"] label span {
    color: var(--text-s) !important;
}

@font-face {
    font-family: 'NanumGothic';
    src: url('file/NanumGothic.ttf') format('truetype');
}
* { font-family: 'NanumGothic', 'Noto Sans KR', sans-serif !important; }

.gradio-container { max-width: 1440px !important; }

/* ── Auth Page ───────────────────────────────────────── */
.auth-wrap {
    max-width: 400px;
    margin: 60px auto;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 36px 32px 28px;
}
.auth-header {
    text-align: center;
    margin-bottom: 28px;
}
.auth-header h2 {
    color: var(--text-p) !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    margin: 0 0 6px 0 !important;
    border: none !important;
}
.auth-header p {
    color: var(--text-m);
    font-size: 12px;
    margin: 0;
}
.auth-msg {
    text-align: center;
    font-size: 13px;
    min-height: 20px;
    margin-top: 6px;
}

/* ── Header ──────────────────────────────────────────── */
.main-header {
    background: var(--header-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 28px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.main-header .hdr-left h1 {
    margin: 0;
    color: var(--text-p) !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border-bottom: none !important;
    letter-spacing: -0.3px;
}
.main-header .hdr-left .sub {
    color: var(--text-m);
    font-size: 11px;
    margin-top: 3px;
}
.main-header .hdr-right {
    display: flex;
    align-items: center;
    gap: 14px;
}
.user-badge {
    background: var(--accent-dim);
    border: 1px solid rgba(96,165,250,0.25);
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
}
.rank-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.4px;
}
.rank-badge.hi { background: var(--green-dim); color: var(--green); border: 1px solid rgba(52,211,153,0.25); }
.rank-badge.lo { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(251,191,36,0.25); }

/* ── Status ──────────────────────────────────────────── */
.status-bar {
    border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    color: var(--text-s);
    background: var(--bg-panel);
    transition: border-left-color 0.3s;
}
.status-bar.idle    { border-left-color: var(--border); color: var(--text-m); }
.status-bar.search  { border-left-color: var(--amber); }
.status-bar.gen     { border-left-color: var(--green); }
.status-bar.done    { border-left-color: var(--accent); }
.status-bar.error   { border-left-color: var(--red); color: var(--red); }

.dot-pulse {
    display: inline-flex; gap: 3px; margin-right: 2px;
}
.dot-pulse span {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.2s ease-in-out infinite;
}
.dot-pulse span:nth-child(2) { animation-delay: 0.2s; }
.dot-pulse span:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1.1); }
}

/* ── Right Panel ─────────────────────────────────────── */
.section-label {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: var(--text-m) !important;
    margin: 12px 0 8px 0 !important;
    padding-bottom: 5px !important;
    border-bottom: 1px solid var(--border) !important;
}
.section-label:first-child { margin-top: 0 !important; }

/* ── Result Summary ──────────────────────────────────── */
.summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 6px;
    margin: 4px 0 8px;
}
.summary-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 4px;
    text-align: center;
}
.summary-val {
    font-size: 16px;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.summary-lbl {
    font-size: 9px;
    color: var(--text-m);
    margin-top: 3px;
}

.info-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid var(--border-light);
    font-size: 11px;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: var(--text-m); }
.info-val { color: var(--text-s); font-weight: 500; }

/* ── Source Documents ────────────────────────────────── */
.source-panel {
    max-height: 260px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}
.source-panel::-webkit-scrollbar { width: 3px; }
.source-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.source-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 5px;
    padding: 7px 10px;
    margin-bottom: 4px;
    font-size: 11px;
    color: var(--text-s);
    word-break: break-all;
    line-height: 1.5;
}
.source-item:last-child { margin-bottom: 0; }
.source-num { color: var(--accent); font-weight: 700; margin-right: 4px; }
.source-score {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 3px;
    margin-left: 4px;
}
.source-empty {
    color: var(--text-m);
    text-align: center;
    padding: 16px 0;
    font-size: 12px;
}

/* ── Query History ───────────────────────────────────── */
.history-list {
    max-height: 340px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}
.history-list::-webkit-scrollbar { width: 3px; }
.history-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.history-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 5px;
    cursor: default;
    transition: background 0.15s;
}
.history-item:hover { background: var(--bg-card-h); }
.history-item:last-child { margin-bottom: 0; }

.history-top {
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
}
.history-user {
    color: var(--accent);
    font-size: 10px;
    font-weight: 600;
}
.history-time {
    color: var(--text-m);
    font-size: 10px;
}
.history-query {
    color: var(--text-p);
    font-size: 12px;
    font-weight: 500;
    line-height: 1.4;
    margin-bottom: 3px;
}
.history-meta {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 10px;
    color: var(--text-m);
}
.check-ico::before { content: ""; color: var(--green); margin-right: 2px; }
.stop-ico::before  { content: ""; color: var(--amber); margin-right: 2px; }

.history-empty {
    color: var(--text-m);
    text-align: center;
    padding: 24px 0;
    font-size: 12px;
}

/* ── Session Tab Bar ────────────────────────────────── */
.session-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 10px;
    padding: 6px 8px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow-x: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}
.session-bar::-webkit-scrollbar { height: 3px; }
.session-bar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.session-tab {
    flex-shrink: 0;
    padding: 5px 14px;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-m);
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
    max-width: 160px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.session-tab:hover { background: var(--bg-card); color: var(--text-s); }
.session-tab.active {
    background: var(--accent-dim);
    border-color: rgba(96,165,250,0.25);
    color: var(--accent);
    font-weight: 600;
}
.session-tab .tab-close {
    margin-left: 6px;
    opacity: 0.4;
    font-size: 10px;
    cursor: pointer;
}
.session-tab .tab-close:hover { opacity: 1; color: var(--red); }
.session-new-btn {
    flex-shrink: 0;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    border: 1px dashed var(--border);
    color: var(--text-m);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.15s;
    background: transparent;
}
.session-new-btn:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-dim);
}
"""


# ── Status HTML Helpers ──────────────────────────────────────────────────────
def _status_idle() -> str:
    return "<div class='status-bar idle'>질문을 입력하면 문서에서 답변을 찾아드립니다.</div>"

def _status_searching() -> str:
    return """<div class='status-bar search'>
        <div class='dot-pulse'><span></span><span></span><span></span></div>
        관련 문서를 검색하고 있습니다...
    </div>"""

def _status_search_done(duration: float, count: int) -> str:
    return f"""<div class='status-bar search' style='border-left-color:var(--green);'>
        {count:,}건의 관련 문서를 찾았습니다. ({duration:.1f}초)
    </div>"""

def _status_generating() -> str:
    return """<div class='status-bar gen'>
        <div class='dot-pulse'><span></span><span></span><span></span></div>
        답변을 생성하고 있습니다...
    </div>"""

def _status_complete(route_dur: float, gen_dur: float) -> str:
    total = route_dur + gen_dur
    return f"<div class='status-bar done'>답변 완료 (총 {total:.1f}초 소요)</div>"

def _status_error(msg: str) -> str:
    return f"<div class='status-bar error'>오류가 발생했습니다: {msg}</div>"


# ── Source HTML ──────────────────────────────────────────────────────────────
def _build_source_html(sources: list) -> str:
    if not sources:
        return "<div class='source-panel'><div class='source-empty'>참조 문서 없음</div></div>"
    html = "<div class='source-panel'>"
    for idx, src in enumerate(sources, 1):
        fpath = src.get("file_path", src) if isinstance(src, dict) else str(src)
        score_val = src.get("score") if isinstance(src, dict) else None
        # 파일명만 표시 (경로 간소화)
        display_name = Path(fpath).name if "/" in fpath else fpath
        score_html = f"<span class='source-score'>관련도 {score_val:.2f}</span>" if score_val is not None else ""
        html += f"<div class='source-item'><span class='source-num'>[{idx}]</span>{display_name}{score_html}</div>"
    html += "</div>"
    return html


# ── Stats HTML ───────────────────────────────────────────────────────────────
def _stats_empty() -> str:
    return "<div class='source-empty'>질의 결과가 여기에 표시됩니다.</div>"

def _stats_html(route_dur: float, gen_dur: float, params: dict, doc_count: int) -> str:
    total = route_dur + gen_dur
    years_str  = ", ".join(str(y) for y in params.get("years",  [])) or "전체 기간"
    months_str = ", ".join(f"{m}월" for m in params.get("months", [])) or "전체"
    search_q   = params.get("search_query", "-") or "-"
    return f"""
<div class='summary-grid'>
  <div class='summary-card'><div class='summary-val'>{doc_count:,}</div><div class='summary-lbl'>검색 문서</div></div>
  <div class='summary-card'><div class='summary-val'>{gen_dur:.1f}s</div><div class='summary-lbl'>답변 생성</div></div>
  <div class='summary-card'><div class='summary-val'>{total:.1f}s</div><div class='summary-lbl'>총 소요</div></div>
</div>
<div class='section-label'>검색 조건</div>
<div class='info-row'><span class='info-key'>기간</span><span class='info-val'>{years_str}</span></div>
<div class='info-row'><span class='info-key'>월</span><span class='info-val'>{months_str}</span></div>
<div class='info-row'><span class='info-key'>검색어</span><span class='info-val'>{search_q}</span></div>
"""


# ── Header HTML ──────────────────────────────────────────────────────────────
def _build_header_html(username: str, user_rank: str) -> str:
    rank_cls = "hi" if user_rank == "hi_rank" else "lo"
    rank_label = "전체 열람" if user_rank == "hi_rank" else "일반 열람"
    return f"""<div class='main-header'>
        <div class='hdr-left'>
            <h1>Legacy Document Intelligence</h1>
            <div class='sub'>9,000건 이상의 레거시 문서를 AI로 검색하고 분석합니다.</div>
        </div>
        <div class='hdr-right'>
            <button class='theme-toggle' onclick='toggleTheme()' title='테마 전환'>🌙</button>
            <span class='rank-badge {rank_cls}'>{rank_label}</span>
            <span class='user-badge'>{username}</span>
        </div>
    </div>"""


# ── Integrated RAG Engine ────────────────────────────────────────────────────
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

    def _log_interaction(self, *, username: str, user_rank: str, query: str,
                         params: dict, filtered_files: list,
                         bm25_references: list, answer: str,
                         route_duration: float, gen_duration: float,
                         status: str = "COMPLETED"):
        try:
            with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
                logs = json.load(f)

            logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": status,
                "user": {
                    "username": username,
                    "rank": user_rank,
                },
                "query": {
                    "raw_input": query,
                    "extracted_parameters": {
                        "years": params.get("years", []),
                        "months": params.get("months", []),
                        "search_query": params.get("search_query", ""),
                    },
                },
                "stage1_routing": {
                    "duration_s": round(route_duration, 3),
                    "filtered_files_count": len(filtered_files),
                    "filtered_files": filtered_files,
                },
                "stage2_generation": {
                    "duration_s": round(gen_duration, 3),
                    "bm25_references": bm25_references,
                    "answer": answer,
                },
                "total_duration_s": round(route_duration + gen_duration, 3),
            })

            with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            logger.info("Interaction logged")
        except Exception as e:
            logger.error(f"Log write fail: {e}")

    def route_query(self, query: str, user_rank: str = "hi_rank"):
        start_t = time.time()
        route_res = self.router.route_query(query, user_rank=user_rank)
        return route_res, time.time() - start_t


# ── Session Helpers ──────────────────────────────────────────────────────────
def _new_session_id():
    return uuid.uuid4().hex[:8]

def _make_sessions_init():
    """초기 세션 데이터 생성"""
    sid = _new_session_id()
    return {sid: {"name": "새 대화", "messages": []}}, sid

def _build_session_bar_html(sessions: dict, active_id: str) -> str:
    """세션 탭 바 HTML 렌더링"""
    html = "<div class='session-bar'>"
    for sid, sdata in sessions.items():
        active_cls = " active" if sid == active_id else ""
        name = sdata.get("name", "대화")
        if len(name) > 18:
            name = name[:16] + "..."
        html += f"<div class='session-tab{active_cls}' data-sid='{sid}'>{name}</div>"
    html += "</div>"
    return html

def _session_display_name(messages: list) -> str:
    """첫 번째 사용자 메시지로 세션 이름 자동 생성"""
    for msg in messages:
        if msg.get("role") == "user":
            q = msg["content"]
            return q if len(q) <= 20 else q[:18] + "..."
    return "새 대화"


# ── Gradio UI ────────────────────────────────────────────────────────────────
def build_gradio_ui():
    engine = IntegratedRAGEngine()

    example_queries = [
        ["2024년 이동장비 주요 정비 내역 요약"],
        ["최근 2년간 엘리베이터 수리 내역"],
        ["교체가 잦은 부품들의 리스트를 보여주세요"],
        ["24년 예방정비 실적"],
        ["트위스트락 교체 이력을 알려줘"],
    ]

    theme_js = """
    function() {
        // Theme toggle logic
        window.toggleTheme = function() {
            const root = document.documentElement;
            const current = root.getAttribute('data-theme') || 'dark';
            const next = current === 'dark' ? 'light' : 'dark';
            root.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
            // Update toggle button icon
            document.querySelectorAll('.theme-toggle').forEach(btn => {
                btn.textContent = next === 'dark' ? '🌙' : '☀️';
            });
        };
        // Apply saved theme on load
        const saved = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', saved);
        // Sync button icons on load
        setTimeout(() => {
            document.querySelectorAll('.theme-toggle').forEach(btn => {
                btn.textContent = saved === 'dark' ? '🌙' : '☀️';
            });
        }, 100);
    }
    """

    with gr.Blocks(css=custom_css, js=theme_js, theme=gr.themes.Base()) as demo:

        user_rank_state = gr.State("low_rank")
        username_state = gr.State("")

        # ── Session State ────────────────────────────────────────────────
        init_sessions, init_active = _make_sessions_init()
        sessions_state = gr.State(init_sessions)       # {sid: {name, messages}}
        active_session_state = gr.State(init_active)    # current session id

        # ════════════════════════════════════════════════════════════════════
        # AUTH VIEW
        # ════════════════════════════════════════════════════════════════════
        with gr.Column(visible=True) as auth_view:
            gr.HTML("""<div class='auth-wrap'>
                <div class='auth-header'>
                    <div style='display:flex;justify-content:flex-end;margin-bottom:8px;'>
                        <button class='theme-toggle' onclick='toggleTheme()' title='테마 전환'>🌙</button>
                    </div>
                    <h2>Legacy Document Intelligence</h2>
                    <p>문서 검색 시스템에 로그인하세요.</p>
                </div>
            </div>""")

            with gr.Column(elem_classes="auth-wrap"):
                with gr.Tab("로그인"):
                    login_user_input = gr.Textbox(label="아이디", placeholder="아이디를 입력하세요")
                    login_pw_input = gr.Textbox(label="비밀번호", type="password", placeholder="비밀번호를 입력하세요")
                    login_btn = gr.Button("로그인", variant="primary")
                    login_msg = gr.HTML("<div class='auth-msg'></div>")

                with gr.Tab("회원가입"):
                    reg_user_input = gr.Textbox(label="아이디", placeholder="사용할 아이디")
                    reg_pw_input = gr.Textbox(label="비밀번호", type="password", placeholder="비밀번호 설정")
                    reg_rank_input = gr.Radio(
                        ["일반 열람 (low_rank)", "전체 열람 (hi_rank)"],
                        label="열람 권한", value="일반 열람 (low_rank)"
                    )
                    reg_btn = gr.Button("회원가입")
                    reg_msg = gr.HTML("<div class='auth-msg'></div>")

        # ════════════════════════════════════════════════════════════════════
        # MAIN VIEW
        # ════════════════════════════════════════════════════════════════════
        with gr.Column(visible=False) as main_view:
            header_display = gr.HTML("")

            # ── 세션 탭 바 ────────────────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=8):
                    session_bar_display = gr.HTML(
                        value=_build_session_bar_html(init_sessions, init_active)
                    )
                with gr.Column(scale=1, min_width=60):
                    new_session_btn = gr.Button("+ 새 대화", size="sm", variant="secondary")
                with gr.Column(scale=1, min_width=60):
                    del_session_btn = gr.Button("삭제", size="sm", variant="stop")

            # ── 세션 선택 드롭다운 (탭 전환용) ────────────────────────────
            session_dropdown = gr.Dropdown(
                choices=[(v["name"], k) for k, v in init_sessions.items()],
                value=init_active,
                label="대화 세션",
                interactive=True,
                container=False,
            )

            with gr.Row():
                # ── 좌측: 채팅 ────────────────────────────────────────────
                with gr.Column(scale=7):
                    chatbot = gr.Chatbot(
                        height=500, show_label=False,
                        placeholder="질문을 입력하면 AI가 문서를 검색하여 답변합니다."
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            show_label=False,
                            placeholder="질문을 입력하세요 (예: 2024년 엘리베이터 정비 내역)",
                            container=False,
                            scale=7,
                        )
                        submit_btn = gr.Button("검색", variant="primary", scale=1)
                        stop_btn   = gr.Button("중지", variant="stop", scale=1, interactive=True)

                    gr.Examples(examples=example_queries, inputs=msg_input, label="자주 묻는 질문")

                # ── 우측: 정보 패널 ───────────────────────────────────────
                with gr.Column(scale=3):
                    status_display = gr.HTML(value=_status_idle())

                    gr.HTML("<div class='section-label'>처리 결과</div>")
                    stats_display = gr.HTML(value=_stats_empty())

                    gr.HTML("<div class='section-label'>참조 문서</div>")
                    source_display = gr.HTML(
                        value="<div class='source-panel'><div class='source-empty'>질의 실행 시 결과가 표시됩니다.</div></div>"
                    )

                    gr.HTML("<div class='section-label'>최근 질문 이력</div>")
                    history_display = gr.HTML(
                        value="<div class='history-empty'>로그인 후 표시됩니다.</div>"
                    )

        # ── State ────────────────────────────────────────────────────────
        msg_query_state = gr.State()
        route_res_state = gr.State()
        route_dur_state = gr.State()

        # ── Helper: dropdown choices 갱신 ────────────────────────────────
        def _dropdown_choices(sessions):
            return [(v["name"], k) for k, v in sessions.items()]

        # ── Event Functions ──────────────────────────────────────────────
        def do_login(user, pw):
            success, msg, rank = login_user(user, pw)
            if success:
                header = _build_header_html(user, rank)
                history = _load_history_html(rank)
                # 로그인 시 새 세션으로 초기화
                sessions, active = _make_sessions_init()
                bar_html = _build_session_bar_html(sessions, active)
                return (gr.update(visible=False), gr.update(visible=True),
                        rank, user,
                        f"<div class='auth-msg' style='color:var(--green);'>{msg}</div>",
                        header, history,
                        sessions, active, bar_html,
                        gr.update(choices=_dropdown_choices(sessions), value=active),
                        [])
            return (gr.update(visible=True), gr.update(visible=False),
                    "low_rank", "",
                    f"<div class='auth-msg' style='color:var(--red);'>{msg}</div>",
                    "", "<div class='history-empty'>로그인 후 표시됩니다.</div>",
                    gr.update(), gr.update(), gr.update(),
                    gr.update(),
                    gr.update())

        login_btn.click(
            do_login,
            inputs=[login_user_input, login_pw_input],
            outputs=[auth_view, main_view, user_rank_state, username_state,
                     login_msg, header_display, history_display,
                     sessions_state, active_session_state, session_bar_display,
                     session_dropdown, chatbot]
        )

        def do_register(user, pw, rank_label):
            rank = "hi_rank" if "hi_rank" in rank_label else "low_rank"
            msg = register_user(user, pw, rank)
            color = "var(--green)" if "완료" in msg else "var(--red)"
            return f"<div class='auth-msg' style='color:{color};'>{msg}</div>"

        reg_btn.click(
            do_register,
            inputs=[reg_user_input, reg_pw_input, reg_rank_input],
            outputs=[reg_msg]
        )

        # ── Session: 새 대화 ─────────────────────────────────────────────
        def create_new_session(sessions, active_id, current_chat):
            # 현재 세션 저장
            if active_id in sessions:
                sessions[active_id]["messages"] = current_chat or []
                sessions[active_id]["name"] = _session_display_name(current_chat or [])
            # 새 세션 생성
            new_id = _new_session_id()
            sessions[new_id] = {"name": "새 대화", "messages": []}
            bar_html = _build_session_bar_html(sessions, new_id)
            return (sessions, new_id, [],
                    bar_html,
                    gr.update(choices=_dropdown_choices(sessions), value=new_id),
                    _status_idle(), _stats_empty(),
                    "<div class='source-panel'><div class='source-empty'>질의 실행 시 결과가 표시됩니다.</div></div>")

        new_session_btn.click(
            create_new_session,
            inputs=[sessions_state, active_session_state, chatbot],
            outputs=[sessions_state, active_session_state, chatbot,
                     session_bar_display, session_dropdown,
                     status_display, stats_display, source_display]
        )

        # ── Session: 삭제 ────────────────────────────────────────────────
        def delete_current_session(sessions, active_id):
            if len(sessions) <= 1:
                # 마지막 세션이면 초기화만
                new_sessions, new_active = _make_sessions_init()
                bar_html = _build_session_bar_html(new_sessions, new_active)
                return (new_sessions, new_active, [],
                        bar_html,
                        gr.update(choices=_dropdown_choices(new_sessions), value=new_active),
                        _status_idle(), _stats_empty(),
                        "<div class='source-panel'><div class='source-empty'>질의 실행 시 결과가 표시됩니다.</div></div>")
            # 삭제 후 인접 세션으로 전환
            sessions.pop(active_id, None)
            new_active = list(sessions.keys())[-1]
            new_chat = sessions[new_active].get("messages", [])
            bar_html = _build_session_bar_html(sessions, new_active)
            return (sessions, new_active, new_chat,
                    bar_html,
                    gr.update(choices=_dropdown_choices(sessions), value=new_active),
                    _status_idle(), _stats_empty(),
                    "<div class='source-panel'><div class='source-empty'>질의 실행 시 결과가 표시됩니다.</div></div>")

        del_session_btn.click(
            delete_current_session,
            inputs=[sessions_state, active_session_state],
            outputs=[sessions_state, active_session_state, chatbot,
                     session_bar_display, session_dropdown,
                     status_display, stats_display, source_display]
        )

        # ── Session: 드롭다운으로 전환 ───────────────────────────────────
        def switch_session(selected_id, sessions, active_id, current_chat):
            if selected_id == active_id:
                return gr.update(), sessions, active_id, gr.update()
            # 현재 세션 저장
            if active_id in sessions:
                sessions[active_id]["messages"] = current_chat or []
                sessions[active_id]["name"] = _session_display_name(current_chat or [])
            # 선택 세션 로드
            new_chat = sessions.get(selected_id, {}).get("messages", [])
            bar_html = _build_session_bar_html(sessions, selected_id)
            return (new_chat, sessions, selected_id, bar_html)

        session_dropdown.change(
            switch_session,
            inputs=[session_dropdown, sessions_state, active_session_state, chatbot],
            outputs=[chatbot, sessions_state, active_session_state, session_bar_display]
        )

        # ── Chat Event Functions ─────────────────────────────────────────
        def user_interaction(user_message, history):
            history = history or []
            history.append({"role": "user", "content": user_message})
            return "", history

        def bot_interaction_route(history, user_rank):
            if not history or history[-1]["role"] != "user":
                yield history, _status_idle(), {}, 0, ""
                return

            if not engine.ready:
                history.append({"role": "assistant", "content": "시스템을 사용할 수 없습니다. 관리자에게 문의하세요."})
                yield history, _status_error("시스템 초기화 실패"), {}, 0, ""
                return

            query = history[-1]["content"]
            history.append({"role": "assistant", "content": "관련 문서를 검색하고 있습니다..."})

            yield history, _status_searching(), {}, 0, query

            route_res, route_duration = engine.route_query(query, user_rank)
            target_count = len(route_res.get("target_files", []))
            yield history, _status_search_done(route_duration, target_count), route_res, route_duration, query

        def bot_interaction_generate(history, route_res, route_duration, query,
                                     username, user_rank,
                                     sessions, active_id):
            if not history or history[-1]["role"] != "assistant":
                yield (history, _build_source_html([]), _stats_empty(),
                       _status_idle(), gr.update(), sessions, active_id,
                       gr.update(), gr.update())
                return

            # 현재 세션의 이전 대화를 chat_history로 전달 (마지막 진행중 메시지 제외)
            chat_history = [m for m in history[:-1] if m.get("content") and
                           m["content"] != "관련 문서를 검색하고 있습니다..."]

            start_gen_t = time.time()
            full_answer = ""
            sources = []
            search_q = route_res.get("search_query", query)

            yield (history, _build_source_html([]), _stats_empty(),
                   _status_generating(), gr.update(), sessions, active_id,
                   gr.update(), gr.update())

            try:
                for rag_chunk in engine.generator.generate_stream(
                    query, route_res["target_files"], search_query=search_q,
                    catalog=engine.router.catalog,
                    params=route_res.get("parameters", {}),
                    chat_history=chat_history,
                ):
                    full_answer = rag_chunk["answer"]
                    sources = rag_chunk["references"]
                    history[-1]["content"] = full_answer
                    yield (history, _build_source_html(sources), _stats_empty(),
                           _status_generating(), gr.update(), sessions, active_id,
                           gr.update(), gr.update())

                gen_dur = time.time() - start_gen_t
                final_stats = _stats_html(route_duration, gen_dur,
                                          route_res.get("parameters", {}),
                                          len(route_res.get("target_files", [])))
                final_status = _status_complete(route_duration, gen_dur)

                engine._log_interaction(
                    username=username, user_rank=user_rank, query=query,
                    params=route_res.get("parameters", {}),
                    filtered_files=route_res.get("target_files", []),
                    bm25_references=sources, answer=full_answer,
                    route_duration=route_duration, gen_duration=gen_dur,
                    status="COMPLETED"
                )

                # 세션에 대화 저장 + 세션명 자동 갱신
                if active_id in sessions:
                    sessions[active_id]["messages"] = history
                    sessions[active_id]["name"] = _session_display_name(history)
                bar_html = _build_session_bar_html(sessions, active_id)
                dd_update = gr.update(choices=_dropdown_choices(sessions), value=active_id)

                updated_history = _load_history_html(user_rank)
                yield (history, _build_source_html(sources), final_stats,
                       final_status, updated_history, sessions, active_id,
                       bar_html, dd_update)

            except Exception as e:
                logger.warning(f"Inference stopped or error: {e}")
                history[-1]["content"] = full_answer + "\n\n[출력이 중단되었습니다.]"
                gen_dur = time.time() - start_gen_t

                engine._log_interaction(
                    username=username, user_rank=user_rank, query=query,
                    params=route_res.get("parameters", {}),
                    filtered_files=route_res.get("target_files", []),
                    bm25_references=sources, answer=history[-1]["content"],
                    route_duration=route_duration, gen_duration=gen_dur,
                    status="STOPPED"
                )

                if active_id in sessions:
                    sessions[active_id]["messages"] = history
                    sessions[active_id]["name"] = _session_display_name(history)
                bar_html = _build_session_bar_html(sessions, active_id)
                dd_update = gr.update(choices=_dropdown_choices(sessions), value=active_id)

                updated_history = _load_history_html(user_rank)
                yield (history, _build_source_html(sources), _stats_empty(),
                       _status_error(str(e)), updated_history, sessions, active_id,
                       bar_html, dd_update)

        # ── Event Chain ──────────────────────────────────────────────────
        _gen_inputs = [chatbot, route_res_state, route_dur_state, msg_query_state,
                       username_state, user_rank_state,
                       sessions_state, active_session_state]
        _gen_outputs = [chatbot, source_display, stats_display, status_display,
                        history_display, sessions_state, active_session_state,
                        session_bar_display, session_dropdown]

        submit_event = (
            submit_btn.click(
                user_interaction,
                [msg_input, chatbot],
                [msg_input, chatbot],
                queue=False,
            )
            .then(
                bot_interaction_route,
                [chatbot, user_rank_state],
                [chatbot, status_display, route_res_state, route_dur_state, msg_query_state],
                queue=True,
                show_progress="hidden",
            )
            .then(
                bot_interaction_generate,
                _gen_inputs,
                _gen_outputs,
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
                [chatbot, user_rank_state],
                [chatbot, status_display, route_res_state, route_dur_state, msg_query_state],
                queue=True,
                show_progress="hidden",
            )
            .then(
                bot_interaction_generate,
                _gen_inputs,
                _gen_outputs,
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
