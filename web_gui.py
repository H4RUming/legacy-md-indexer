import logging
import gradio as gr
from agentic_router import AgenticRouter
from rag_generator import RAGGenerator

# 전역 로거
logger = logging.getLogger("RAG_UI")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s', '%H:%M:%S'))
logger.addHandler(ch)

# 로컬 폰트(NanumGothic) 및 커스텀 UI 디자인 적용 (HTML 주입용)
custom_css = """
@font-face {
    font-family: 'NanumGothic';
    src: url('file/NanumGothic.ttf') format('truetype');
}
* { font-family: 'NanumGothic', sans-serif !important; }
.gradio-container { max-width: 1400px !important; }
.header-banner { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 24px 32px; border-radius: 8px; color: white; margin-bottom: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
.header-banner h1 { margin: 0; color: #f8fafc; font-weight: 700; font-size: 24px; letter-spacing: -0.5px; }
.header-banner p { margin: 8px 0 0 0; color: #cbd5e1; font-size: 14px; }
.source-panel { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 16px; font-size: 13px; max-height: 550px; overflow-y: auto; }
.source-item { margin-bottom: 8px; padding: 10px 12px; background: white; border-left: 4px solid #475569; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); word-break: break-all; color: #334155; }
"""

class IntegratedRAGEngine:
    """백엔드 모듈 연동 래퍼"""
    def __init__(self):
        try:
            self.router = AgenticRouter(catalog_path="./file_catalog.json")
            self.generator = RAGGenerator(target_dir="./processed_md")
            logger.info("RAG 엔진 로드 완료")
        except Exception as e:
            logger.error(f"엔진 초기화 에러: {e}")

    def execute_query(self, query: str):
        route_res = self.router.route_query(query)
        rag_res = self.generator.generate(
            query=route_res["query"], 
            target_files=route_res["target_files"]
        )
        return rag_res["answer"], rag_res["references"], route_res["parameters"]

def build_gradio_ui():
    engine = IntegratedRAGEngine()
    
    example_queries = [
        ["2024년 이동장비 주요 정비 내역 요약"],
        ["최근 2년간 엘리베이터 수리 내역"],
        ["어제 진행된 정비 로그에서 특이사항 정리해 줘"],
        ["교체시기가 오래된 부품들의 리스트를 알려줘"],
        ["교체가 잦은 부품들의 리스트를 보여주세요"],
        ["24년 예방정비 실적"],
        ["CM2350 Technical Package에 대해 설명해줘"],
        ["크레인 연간 정비 계획"],
        ["트위스트락 교체 이력을 알려줘"],
        ["COSCO Shipping 업무협의 내용"]
    ]

    # Warning 방지를 위해 Blocks의 css 파라미터 제거
    with gr.Blocks() as demo:
        # CSS를 HTML 내부에 직접 주입하여 안정성 확보
        gr.HTML(f"""
        <style>{custom_css}</style>
        <div class="header-banner">
            <h1>LLM-Based Hierarchical Document Intelligence Engine</h1>
            <p>현재 프로토타입 버전으로, 불안정하거나 간혹 잘못된 정보를 제공할 수 있습니다.</p>
        </div>
        """)
        
        with gr.Row():
            # 왼쪽: 챗봇 UI
            with gr.Column(scale=7):
                # TypeError 방지를 위해 type="messages" 제거
                chatbot = gr.Chatbot(height=600, show_label=False)
                with gr.Row():
                    msg_input = gr.Textbox(
                        show_label=False, 
                        placeholder="질의를 입력하십시오.", 
                        container=False, 
                        scale=8
                    )
                    submit_btn = gr.Button("실행", variant="primary", scale=1)
                
                gr.Examples(
                    examples=example_queries,
                    inputs=msg_input,
                    label="자주 묻는 질문 예시"
                )
                
                gr.ClearButton([msg_input, chatbot], value="세션 초기화", size="sm")

            # 오른쪽: 상태 및 참조 패널
            with gr.Column(scale=3):
                gr.Markdown("### 시스템 아키텍처 상태")
                with gr.Group():
                    gr.Textbox(value="qwen2.5:14b (Local Ollama)", label="추론 엔진", interactive=False)
                    gr.Textbox(value="Agentic Router + MD RAG", label="검색 아키텍처", interactive=False)
                
                # 라우터가 추출한 파라미터 확인용
                gr.Markdown("### Optimized Query Process")
                params_display = gr.JSON(label="추출된 검색 파라미터", value={})
                
                gr.Markdown("### 참조 문서 데이터 원천 (Sources)")
                source_display = gr.HTML(value="<div class='source-panel' style='color: #64748b; text-align: center; padding-top: 30px;'>질의 실행 시 참조된 원본 문서 경로가 표출됩니다.</div>")

        # 튜플 리스트 방식 [[user, bot]]으로 상태 관리 변경
        def user_interaction(user_message, history):
            history = history or []
            history.append([user_message, None])
            return "", history

        def bot_interaction(history):
            if not history: 
                return history, "", {}
            
            raw_user_message = history[-1][0]
            answer, sources, params = engine.execute_query(raw_user_message)
            
            source_html = "<div class='source-panel'>"
            if sources:
                for idx, src in enumerate(sources, 1): 
                    source_html += f"<div class='source-item'><b>[{idx}]</b> {src}</div>"
            else: 
                source_html += "<div style='color: #64748b; text-align: center; padding: 20px;'>참조된 문서 데이터가 존재하지 않습니다.</div>"
            source_html += "</div>"
            
            history[-1][1] = answer
            return history, source_html, params

        # 이벤트 체인
        msg_input.submit(user_interaction, [msg_input, chatbot], [msg_input, chatbot], queue=False).then(
            bot_interaction, chatbot, [chatbot, source_display, params_display]
        )
        submit_btn.click(user_interaction, [msg_input, chatbot], [msg_input, chatbot], queue=False).then(
            bot_interaction, chatbot, [chatbot, source_display, params_display]
        )
        
    return demo

if __name__ == "__main__":
    logger.info("웹 서버 시작 (Share 활성화)")
    app = build_gradio_ui()
    # share 켜고 폰트 로드를 위해 allowed_paths 등록
    app.launch(server_name="0.0.0.0", server_port=7860, share=True, allowed_paths=["."])