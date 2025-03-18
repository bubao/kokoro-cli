import soundfile as sf
import sys
import os
import subprocess
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
import torch
from tqdm import tqdm
from kokoro import KModel, KPipeline
import csv

# 解决 PyTorch 警告
import warnings
import jieba
import logging

jieba.setLogLevel(logging.INFO)


with open("voices.csv", "r") as f:
    reader = csv.DictReader(f)
    # 获取所有中文女性语音文件
    female_zh_voices = [
        row["文件名"]
        for row in reader
        if row["性别"] == "female" and row["语言"] == "zh"
    ]

# 在代码开头添加以下内容
warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated",
    category=FutureWarning,
    module="torch.nn.utils.weight_norm",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="dropout option adds dropout after all but last recurrent layer",
)

try:
    import readline  # Linux/macOS
except ImportError:
    import pyreadline as readline  # Windows


class KokoroTTS:
    def __init__(
        self,
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        sample_rate=24000,
        n_zeros=5000,
        join_sentences=True,
        voice="zf_001",
        device=None,
    ):
        self.repo_id = repo_id
        self.sample_rate = sample_rate
        self.n_zeros = n_zeros
        self.join_sentences = join_sentences
        self.voice = voice
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型和管道
        self.model = KModel(repo_id=self.repo_id).to(self.device).eval()
        self.en_pipeline = KPipeline(lang_code="a", repo_id=self.repo_id, model=False)

        # 定义发音转换函数
        self.en_callable = lambda text: (
            "kˈOkəɹO"
            if text == "Kokoro"
            else "sˈOl" if text == "Sol" else next(self.en_pipeline(text)).phonemes
        )

        # 定义速度控制函数
        self.speed_callable = lambda len_ps: (
            (1 - max(0, (len_ps - 83) / 500)) * 0.8 * 1.1
            if 83 < len_ps < 183
            else 0.8 * 1.1
        )

    def generate_audio(self, texts, output_path="output.wav", save_intermediate=False):
        path = Path(output_path).parent.resolve()
        path.mkdir(parents=True, exist_ok=True)

        wavs = []
        for paragraph in tqdm(texts, desc="Processing paragraphs"):
            if self.join_sentences and len(paragraph) > 1:
                paragraph = ["".join(paragraph)]

            for i, sentence in enumerate(paragraph):
                generator = KPipeline(
                    lang_code="z",
                    repo_id=self.repo_id,
                    model=self.model,
                    en_callable=self.en_callable,
                )(sentence, voice=self.voice, speed=self.speed_callable)

                audio_result = next(generator)
                wav = audio_result.audio

                if save_intermediate:
                    sf.write(path / f"zh_{len(wavs):02d}.wav", wav, self.sample_rate)

                if i == 0 and wavs and self.n_zeros > 0:
                    wav = np.concatenate([np.zeros(self.n_zeros), wav])

                wavs.append(wav)

        final_audio = np.concatenate(wavs)
        sf.write(output_path, final_audio, self.sample_rate)
        return output_path

    def set_voice(self, voice_name):
        self.voice = voice_name

    def set_speed_function(self, speed_func):
        self.speed_callable = speed_func

    def set_phoneme_function(self, phoneme_func):
        self.en_callable = phoneme_func


class InteractiveKokoroTTS(KokoroTTS):
    def generate_audio_data(self, texts):
        wavs = []
        for paragraph in texts:
            for i, sentence in enumerate(paragraph):
                generator = self._create_generator(sentence)
                result = next(generator)
                wav = result.audio
                if i == 0 and wavs and self.n_zeros > 0:
                    wav = np.concatenate([np.zeros(self.n_zeros), wav])
                wavs.append(wav)
        return np.concatenate(wavs)

    def _create_generator(self, sentence):
        return KPipeline(
            lang_code="z",
            repo_id=self.repo_id,
            model=self.model,
            en_callable=self.en_callable,
        )(sentence, voice=self.voice, speed=self.speed_callable)


def play_audio(audio_data, sample_rate):
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, audio_data, sample_rate)
        temp_path = temp_file.name

    try:
        if sys.platform == "linux":
            subprocess.run(["aplay", "-q", temp_path], check=True)
        elif sys.platform == "darwin":
            subprocess.run(["afplay", temp_path], check=True)
        elif sys.platform == "win32":
            subprocess.run(["start", " ", temp_path], shell=True, check=True)
        else:
            print("不支持的平台，无法播放音频")
    except subprocess.CalledProcessError:
        print("播放失败，请确保系统音频播放器可用")
    finally:
        os.remove(temp_path)


def start_interactive_mode():
    tts = InteractiveKokoroTTS(join_sentences=True, voice="zf_001")

    available_commands = ["/exit", "/help", "/show", "/set", "/save"]

    def custom_completer(text, state):
        responses = []
        for cmd in available_commands:
            if cmd.startswith(text):
                responses.append(cmd)
        if state < len(responses):
            return responses[state]
        return None

    # 设置补全
    if sys.platform == "win32":
        readline.parse_and_bind("tab: complete")
    else:
        readline.parse_and_bind("tab: complete")  # 修正绑定语法

    readline.set_completer(custom_completer)

    print("进入Kokoro TTS交互模式")
    print("输入文本直接生成音频，以 / 开头为命令")
    print("输入 /help 查看帮助\n")

    while True:
        try:
            user_input = input("kokoro> ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                handle_system_command(user_input[1:], tts)
            else:
                generate_and_play(user_input, tts)

        except KeyboardInterrupt:
            print("\n强制退出交互模式")
            break
        except Exception as e:
            print(f"发生错误：{str(e)}")


def handle_system_command(cmd, tts):
    parts = cmd.split()
    command = parts[0] if parts else ""

    if command == "exit":
        print("退出交互模式")
        sys.exit(0)
    elif command == "help":
        print_help()
    elif command == "show":
        print_current_settings(tts)
    elif command == "set":
        handle_set_command(" ".join(parts[1:]), tts)
    elif command == "save":
        if len(parts) < 2:
            print("用法：/save <文件路径>")
        else:
            output_path = parts[1]
            try:
                if hasattr(tts, "last_audio"):
                    sf.write(output_path, tts.last_audio, tts.sample_rate)
                    print(f"音频已保存到 {output_path}")
                else:
                    print("尚未生成音频，无法保存")
            except Exception as e:
                print(f"保存失败：{str(e)}")
    else:
        print(f"未知命令：{command}")


def handle_set_command(cmd_str, tts):
    try:
        param, value = cmd_str.split(" ", 1)
    except ValueError:
        print("用法：set <参数> <值>")
        return

    try:
        if hasattr(tts, param):
            current_value = getattr(tts, param)
            if isinstance(current_value, bool):
                new_val = value.lower() in ["true", "1", "yes"]
            elif isinstance(current_value, int):
                new_val = int(value)
            else:
                new_val = value
            setattr(tts, param, new_val)
            print(f"设置成功：{param} = {new_val}")
        else:
            print(f"无效参数：{param}")
    except Exception as e:
        print(f"设置失败：{str(e)}")


def generate_and_play(text, tts):
    if not text.strip():
        # print("请输入有效文本")
        return

    paragraphs = [[text.strip()]]
    try:
        audio_data = tts.generate_audio_data(paragraphs)
        play_audio(audio_data, tts.sample_rate)
        # 记录最近生成的音频数据（供保存功能使用）
        tts.last_audio = audio_data  # 新增：保存音频数据到实例属性
    except Exception as e:
        print(f"生成失败：{str(e)}")


def print_current_settings(tts):
    print("当前配置：")
    print(f"  声音: {tts.voice}")
    print(f"  合并句子: {tts.join_sentences}")
    print(f"  采样率: {tts.sample_rate}")
    print(f"  静音间隔: {tts.n_zeros} 样本点")
    print(f"  设备: {tts.device}")


def print_help():
    help_msg = """
可用命令：
/exit           退出程序
/help           显示帮助
/show           显示当前配置
/set <参数> <值> 设置参数（如：set voice zm_010）
/save <路径>    保存最近生成的音频

输入文本直接生成音频：
例如：你好，Kokoro！
"""
    print(help_msg)


def main():
    if len(sys.argv) == 1:
        start_interactive_mode()
    else:
        pass


if __name__ == "__main__":
    main()
