import logging
import gradio as gr
from agentic_router import AgenticRouter
from rag_generator import RAGGenerator

logger = logging.getLogger("RAG_GUI")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s', '%H:%M:%S'))
logger.addHandler(ch)

# Initialize Backend Modules
try:
    router = AgenticRouter(catalog_path="./file_catalog.json")
    generator = RAGGenerator(target_dir="./processed_md")
    logger.info("Backend modules initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize backend modules: {e}")

def process_query(query: str):
    logger.info(f"Received query: {query}")
    
    # 1. Routing & Parameter Extraction
    route_result = router.route_query(query)
    optimized_params = route_result["parameters"]
    target_files = route_result["target_files"]
    logger.info(f"Optimized parameters: {optimized_params}")
    
    # 2. RAG Generation
    rag_result = generator.generate(query=query, target_files=target_files)
    logger.info("RAG generation completed.")
    
    answer = rag_result["answer"]
    references = "\n".join([f"- {ref}" for ref in rag_result["references"]])
    
    if not references:
        references = "No references found."
        
    return optimized_params, answer, references

# Gradio 5.x Blocks Interface
with gr.Blocks(title="Legacy Data RAG System", theme=gr.themes.Default()) as app:
    gr.Markdown("## Legacy Document RAG Search Interface")
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Search Query", 
                placeholder="Enter your natural language query here...", 
                lines=3
            )
            submit_btn = gr.Button("Search", variant="primary")
            
            gr.Markdown("### Routing Process")
            params_output = gr.JSON(label="Optimized Query Parameters")
            
        with gr.Column(scale=2):
            answer_output = gr.Markdown(label="AI Answer")
            ref_output = gr.Textbox(label="Referenced Files", lines=6, interactive=False)

    submit_btn.click(
        fn=process_query,
        inputs=query_input,
        outputs=[params_output, answer_output, ref_output]
    )
    
    query_input.submit(
        fn=process_query,
        inputs=query_input,
        outputs=[params_output, answer_output, ref_output]
    )

if __name__ == "__main__":
    logger.info("Starting Gradio web server...")
    app.launch(server_name="0.0.0.0", server_port=7860)