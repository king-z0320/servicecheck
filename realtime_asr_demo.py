#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎤 实时语音识别演示 (FunASR Streaming) - Debug版

功能:
- 自动清理端口占用
- 强制使用 paraformer-zh-streaming
- 详细打印每一步的音频状态、Cache变化和模型输出
"""

import os
import sys
import subprocess
import signal
import numpy as np
import gradio as gr
import tempfile
import time
import json
from datetime import datetime

# 全局变量
asr_model_streaming = None
asr_model_offline = None
emotion_model = None

def kill_port(port):
    """自动杀掉占用端口的进程"""
    try:
        # 查找占用端口的 PID
        cmd = f"lsof -ti:{port}"
        pid = subprocess.check_output(cmd, shell=True).decode().strip()
        if pid:
            print(f"⚠️ 发现端口 {port} 被进程 {pid} 占用，正在清理...")
            os.kill(int(pid), signal.SIGKILL)
            print(f"✅ 进程 {pid} 已已清理")
            time.sleep(1) # 等待释放
    except subprocess.CalledProcessError:
        # 没有进程占用
        pass
    except Exception as e:
        print(f"❌ 清理端口失败: {e}")

def load_models():
    """加载模型"""
    global asr_model_streaming, asr_model_offline, emotion_model
    import torch
    from funasr import AutoModel
    
    # 检测设备
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"🖥️ 使用设备: {device}")
    
    # 1. 加载流式 ASR 模型
    print("⏳ 正在加载流式 ASR 模型 (paraformer-zh-streaming)...")
    try:
        asr_model_streaming = AutoModel(
            model="paraformer-zh-streaming",
            device=device,
        )
        print("✅ 流式 ASR 模型加载完成")
    except Exception as e:
        print(f"❌ 流式模型加载失败: {e}")

    # 2. 加载离线 ASR 模型
    print("⏳ 正在加载离线 ASR 模型 (paraformer-zh)...")
    try:
        asr_model_offline = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            device=device,
        )
        print("✅ 离线 ASR 模型加载完成")
    except Exception as e:
        print(f"❌ 离线模型加载失败: {e}")
    
    # 3. 加载情感识别模型
    print("⏳ 正在加载情感识别模型 (emotion2vec)...")
    try:
        emotion_model = AutoModel(
            model="iic/emotion2vec_plus_large",
            device=device,
        )
        print("✅ 情感识别模型加载完成")
    except Exception as e:
        print(f"⚠️ 情感模型加载失败: {e}")

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"

def transcribe_realtime(audio_path, state):
    """
    伪流式转录 (Pseudo-Streaming)
    原理：持续累积音频，每次截取最近的 N 秒（滑动窗口）送入离线模型识别。
    优势：规避了流式模型 Cache 管理的复杂性，利用离线模型的强大能力，只要计算速度够快，体验就是实时的。
    """
    global asr_model_offline
    
    # 初始化状态
    if state is None:
        state = {
            "full_text_history": "", # 已固定的历史文本
            "audio_buffer": np.array([], dtype=np.float32), # 当前未固定的音频缓冲
            "start_time": time.time(),
            "last_rec_time": time.time(),
            "call_idx": 0
        }
    
    if audio_path is None:
        return "🎙️ 准备就绪，请点击录音...", state
    
    state["call_idx"] += 1
    idx = state["call_idx"]
    
    try:
        import soundfile as sf
        # 读取本次音频片段
        audio_data, sample_rate = sf.read(audio_path)
        
        # 预处理：转单声道、重采样到 16k
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        if sample_rate != 16000:
            import scipy.signal
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = scipy.signal.resample(audio_data, num_samples)
        
        # 追加到缓冲区
        state["audio_buffer"] = np.concatenate((state["audio_buffer"], audio_data))
        
        # 策略：
        # 每次识别最近的 15 秒音频。
        # 为什么是 15 秒？太长会慢，太短上下文不够。
        # 我们不进行"固定"操作，而是每次都刷新显示"最近听到的内容"。
        # 这种方式对于 Demo 演示最简单粗暴且有效。
        
        MAX_WINDOW_SECONDS = 15
        max_samples = MAX_WINDOW_SECONDS * 16000
        
        # 取最近窗口
        current_window = state["audio_buffer"]
        if len(current_window) > max_samples:
           current_window = current_window[-max_samples:]
           
        # 音量检测（简单的静音门限，避免静音也疯狂识别）
        if np.max(np.abs(audio_data)) < 0.01 and len(state["audio_buffer"]) % 16000 != 0:
             # 如果这帧是静音，且没有凑够整秒，稍微偷懒一下？不，为了流畅还是算吧
             pass

        # 保存为临时文件送给离线模型
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, current_window, 16000)
            
        try:
            if asr_model_offline:
                t0 = time.time()
                # 使用离线模型识别
                res = asr_model_offline.generate(input=temp_path)
                t1 = time.time()
                
                text = ""
                if res and len(res) > 0:
                    text = res[0].get("text", "").strip()
                
                print(f"[{idx}] � 识别窗口({len(current_window)/16000:.1f}s) 耗时{(t1-t0)*1000:.0f}ms -> {text}")
                
                # 为了让显示像字幕一样：
                # 我们简单地显示当前窗口识别出的所有内容。
                # 随着说话，内容会变长，然后旧的会被移出窗口。
                # 这是一个"滑动字幕"效果。
                
                state["current_text"] = text
                
            else:
                 print("Error: Offline model not loaded")
                 
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    
    # 格式化显示
    # 显示最近 15 秒识别到的完整文本
    output = f"🎙️ 实时转录 (Pseudo-Streaming) \n"
    output += "─" * 40 + "\n\n"
    
    txt = state.get("current_text", "")
    if txt:
        # 简单断句显示
        # 每 20 个字换行
        lines = [txt[i:i+20] for i in range(0, len(txt), 20)]
        for line in lines:
            output += f"� {line}\n"
    else:
        output += "Waiting for speech..."
        
    output += "\n" + "─" * 40
    return output, state

def format_display(state, current_time):
    # 此函数在伪流式中不再使用，逻辑合并到了 transcribe_realtime
    return ""

def reset_state():
    return "🎙️ 已重置", None

def transcribe_final(audio):
    """录音后分析"""
    if audio is None: return "⚠️ 请先录制音频", "", ""
    import soundfile as sf
    sample_rate, audio_data = audio
    # 简单处理...
    if len(audio_data.shape) > 1: audio_data = audio_data.mean(axis=1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, audio_data, sample_rate)
    try:
        text = "(模型未加载)"
        if asr_model_offline:
            res = asr_model_offline.generate(input=temp_path)
            text = res[0].get("text", "") if res else ""
        
        emo_txt = ""
        if emotion_model and text:
            time.sleep(0.1) # 简单模拟
            emo_res = emotion_model.generate(input=temp_path, granularity="utterance")
            if emo_res:
                labels = emo_res[0].get("labels", [])
                scores = emo_res[0].get("scores", [])
                sorted_pairs = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)[:3]
                emo_txt = "\n".join([f"{l}: {s:.2%}" for l, s in sorted_pairs])
        
        return text, emo_txt, f"时长: {len(audio_data)/sample_rate:.1f}s"
    finally:
        if os.path.exists(temp_path): os.unlink(temp_path)

def create_demo():
    with gr.Blocks(title="ASR Debug Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎤 ASR 实时流式调试版")
        gr.Markdown("请查看终端输出以获取详细调试信息。")
        
        with gr.Tabs():
            with gr.Tab("🔴 实时流式"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 开启 share=True 会生成 HTTPS 链接，解决浏览器麦克风权限问题
                        audio = gr.Audio(sources=["microphone"], type="filepath", streaming=True, label="说话...")
                        reset_btn = gr.Button("重置")
                    with gr.Column(scale=2):
                        output = gr.Textbox(label="实时结果", lines=10, interactive=False)
                state = gr.State()
                audio.stream(fn=transcribe_realtime, inputs=[audio, state], outputs=[output, state])
                reset_btn.click(fn=reset_state, outputs=[output, state])
            
            with gr.Tab("⏺️ 录音后分析"):
                audio_batch = gr.Audio(sources=["microphone", "upload"], type="numpy")
                btn = gr.Button("识别")
                out1 = gr.Textbox(label="文本")
                out2 = gr.Textbox(label="情感")
                out3 = gr.Textbox(label="信息")
                btn.click(fn=transcribe_final, inputs=[audio_batch], outputs=[out1, out2, out3])
    return demo

if __name__ == "__main__":
    # 1. 自动杀端口
    kill_port(7860)
    
    # 2. 加载模型
    load_models()
    
    # 3. 启动服务 (开启 share=True 以绕过麦克风限制)
    print("🚀 正在启动服务...")
    print("⚠️ 注意: 如果本地浏览器麦克风无法访问，请使用控制台输出的 public URL (https://....gradio.live)")
    create_demo().launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=True 
    )
