from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from .config import DEMO_MODELS, OUTPUT_DIR
from .json_utils import dumps_pretty
from .ollama_client import OllamaClient
from .pipeline import SextuplePipeline


DEFAULT_DIALOGUE = """张三：昨晚住的酒店一开始让我很失望，房间有烟味，空调也有点吵。
李四：你后来找前台了吗？
张三：找了，前台十分钟内就给我换了房，还主动送了早餐券，服务态度很好。
李四：那你现在怎么看这家酒店？
张三：房间卫生需要改进，但前台处理问题很积极，我对服务已经变满意了。"""


def run_demo(dialogue: str, model: str, mode: str) -> tuple[str, pd.DataFrame, pd.DataFrame, str, str]:
    pipeline = SextuplePipeline()
    try:
        result = pipeline.run(dialogue, model=model, doc_id="web_demo", mode=mode)
        sextuples = pd.DataFrame(result.get("sextuples", []))
        flips = pd.DataFrame(result.get("sentiment_flips", []))
        status = (
            f"valid_json={result.get('valid_json')} | "
            f"model={result.get('model')} | "
            f"latency={result.get('latency_seconds')}s | "
            f"mode={result.get('mode')}"
        )
        export_path = export_result(result)
        return status, sextuples, flips, dumps_pretty(result), str(export_path)
    except Exception as exc:
        message = (
            "Ollama API call failed. Run `python -m app.cli check-ollama`, "
            "start Ollama, or select `heuristic-demo` for local interface validation. "
            f"Error: {exc}"
        )
        return message, pd.DataFrame(), pd.DataFrame(), "{}", ""


def export_result(result: dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "web_demo_result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def ollama_status() -> str:
    client = OllamaClient()
    available = client.is_available()
    models = client.list_models()
    return dumps_pretty({"available": available, "models": models})


def default_model() -> str:
    installed = set(OllamaClient().list_models())
    for candidate in ["qwen2.5:7b", "qwen2.5:3b", "llama3.1:8b", "heuristic-demo"]:
        if candidate == "heuristic-demo" or candidate in installed:
            return candidate
    return "heuristic-demo"


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Ollama 情感六元组抽取系统") as demo:
        gr.Markdown("# Ollama 多轮对话情感六元组抽取")
        with gr.Row():
            model = gr.Dropdown(DEMO_MODELS, value=default_model(), label="模型")
            mode = gr.Radio(["four-step", "one-shot"], value="four-step", label="Prompt 策略")
        dialogue = gr.Textbox(value=DEFAULT_DIALOGUE, lines=8, label="输入对话")
        with gr.Row():
            run_button = gr.Button("运行抽取", variant="primary")
            status_button = gr.Button("检查 Ollama")
        status = gr.Textbox(label="运行状态", lines=2)
        sextuples = gr.Dataframe(label="情感六元组", interactive=False)
        flips = gr.Dataframe(label="情感转折", interactive=False)
        raw_json = gr.Code(label="完整 JSON", language="json")
        export_path = gr.Textbox(label="导出路径")

        run_button.click(run_demo, inputs=[dialogue, model, mode], outputs=[status, sextuples, flips, raw_json, export_path])
        status_button.click(ollama_status, outputs=status)
    return demo


def main() -> None:
    create_demo().launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
