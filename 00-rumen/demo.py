import os
import gradio as gr
from transformers import pipeline

# 设置代理环境变量
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"

# 加载模型
qa_pipeline = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa", device=0)

# 清除代理环境变量
del os.environ['HTTP_PROXY']
del os.environ['HTTPS_PROXY']

# 定义 Gradio 接口
def qa_function(context, question):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

iface = gr.Interface(fn=qa_function, inputs=["text", "text"], outputs="text")

# 启动 Gradio 接口，不使用代理
iface.launch(share=True)