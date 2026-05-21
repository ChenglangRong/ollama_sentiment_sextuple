# Ollama 多轮对话情感六元组抽取课程设计

## 快速运行

```powershell
cd "C:\Users\Lenovo\Desktop\大三\大三下\自然语言处理\自然语言处理-课程设计\ollama_sentiment_sextuple"
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\scripts\setup_ollama.ps1
python app.py
```

浏览器打开 Gradio 显示的本地地址，默认端口为 `http://127.0.0.1:7860`。

## CLI

```powershell
python -m app.cli check-ollama
python -m app.cli infer --model qwen2.5:7b --input data/examples.jsonl --output outputs/predictions.jsonl
python -m app.cli evaluate --models qwen2.5:7b llama3.1:8b qwen2.5:3b --limit 120
python -m app.cli evaluate-flips --model qwen2.5:7b
python -m pytest -q
```

模型下载完成前可用 `heuristic-demo` 验证界面、JSON schema 和评测文件生成：

```powershell
python -m app.cli evaluate --models heuristic-demo --limit 30 --dataset examples
python app.py
```

## 目录

- `app/`：Ollama 客户端、prompt 流水线、评测、Gradio 页面。
- `data/`：课程演示样例、公开评测备用样例、30 条情感转折金标样例。
- `outputs/`：评测结果、预测结果、Web 导出结果。
- `docs/`：课程设计报告、答辩讲稿、评分点对齐表。
- `scripts/`：Ollama 部署、完整评测、报告 docx 生成脚本。

## 评分点对应

- 报告完整性：`docs/课程设计报告.md` 与 `docs/课程设计报告.docx`。
- 任务定义与方案设计：四步链式情感六元组抽取，含任务边界、输入输出、替代方案。
- 系统实现：`python app.py` 和 `python -m app.cli ...` 双入口。
- 效果评测：公开 ASTE 数据集、模型对比、prompt 策略对比、30 条情感转折评测。
- 答辩加分：`docs/答辩讲稿.md` 和 `docs/评分点对齐表.md`。

## 最终提交包

最终提交包为 `ollama_sentiment_sextuple_final.zip`。压缩包包含代码、数据、脚本、测试、报告、答辩材料、正式评测 CSV、中文演示 JSON 和 Gradio 截图；Ollama 模型权重保留在本机 `D:\JZDSTool\ollama_models`，通过 `scripts/setup_ollama.ps1` 复现部署。
