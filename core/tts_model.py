# core/tts_model.py
import torch
import numpy as np
from kokoro import KModel, KPipeline
import warnings
from config.voice_config import VoiceConfigManager  # 使用动态配置管理
from config.command_config import CommandConfig  # 引入命令配置


class InteractiveKokoroTTS:
    PARAM_DESCRIPTIONS = {
        "voice": "选择声优",
        "join_sentences": "是否将句子连接在一起",
        "sample_rate": "音频采样率",
        "n_zeros": "句子之间的静音长度",
        "device": "运行设备（CPU 或 GPU）",
    }

    def __init__(
        self,
        repo_id=None,
        voice=None,  # 初始默认值需动态获取
        sample_rate=24000,
        n_zeros=5000,
        join_sentences=True,
        device=None,
        command_config=None,  # 接收CommandConfig实例
    ):
        self.model = None
        self.en_pipeline = None
        self._current_repo_id = None
        self.command_config = (
            command_config or CommandConfig()
        )  # 确保command_config是实例
        self._voice_config = VoiceConfigManager()
        self._command_config = self.command_config.get_config()
        self._sample_rates = self._command_config["param_values"]["sample_rate"]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.last_audio = None  # 新增属性
        self._sample_width = 2  # 假设样本位宽为 16 位（2 字节）

        # 初始化警告过滤
        warnings.filterwarnings("ignore", category=UserWarning, message="dropout.*")

        # 初始化参数
        if repo_id is not None:
            self._validate_and_set_params(
                repo_id=repo_id,
                voice=voice,
                sample_rate=sample_rate,
                n_zeros=n_zeros,
                join_sentences=join_sentences,
            )
        else:
            self._voice = voice
            self._sample_rate = sample_rate
            self._n_zeros = n_zeros
            self._join_sentences = join_sentences

    def _validate_and_set_params(self, **kwargs):
        """参数校验与设置"""
        for key, value in kwargs.items():
            if key == "repo_id":
                self._set_repo_id(value)
            elif key == "voice":
                self._set_voice(value)
            elif key == "sample_rate":
                self._set_sample_rate(value)
            elif key == "n_zeros":
                self._set_n_zeros(value)
            elif key == "join_sentences":
                self._set_join_sentences(value)
            else:
                raise ValueError(f"无效参数: {key}")

    def _set_repo_id(self, repo_id):
        if (
            repo_id
            and repo_id
            not in self.command_config.get_config()["commands"][".load"]["params"][0][
                "choices"
            ]
        ):
            raise ValueError(
                f"无效模型ID: {repo_id}. 可用模型：{self.command_config.get_config()['commands']['.load']['params'][0]['choices']}"
            )
        self._current_repo_id = repo_id
        self._load_model(repo_id)

    def _set_voice(self, voice):
        if not self._current_repo_id:
            raise ValueError("请先加载模型以选择声优")
        valid_voices = [
            v["name"]
            for v in self.command_config.get_voice_options(self._current_repo_id)
        ]
        if voice not in valid_voices:
            raise ValueError(
                f"无效声优: {voice}. 可用声优：{valid_voices}（模型：{self._current_repo_id}）"
            )
        self._voice = voice

    def _set_sample_rate(self, sample_rate):
        if str(sample_rate) not in self._command_config["param_values"]["sample_rate"]:
            raise ValueError(
                f"无效采样率: {sample_rate}. 允许的值：{self._command_config['param_values']['sample_rate']}"
            )
        self._sample_rate = int(sample_rate)

    def _set_n_zeros(self, n_zeros):
        if str(n_zeros) not in self._command_config["param_values"]["n_zeros"]:
            raise ValueError(
                f"无效静音长度: {n_zeros}. 允许的值：{self._command_config['param_values']['n_zeros']}"
            )
        self._n_zeros = int(n_zeros)

    def _set_join_sentences(self, join_sentences):
        if not isinstance(join_sentences, bool):
            raise TypeError("join_sentences 必须为布尔值")
        self._join_sentences = join_sentences

    @property
    def available_models(self):
        """返回所有可用模型"""
        return list(
            self.command_config.get_config()["commands"][".load"]["params"][0][
                "choices"
            ]
        )

    @property
    def available_voices(self):
        """返回当前模型的可用声优"""
        if not self._current_repo_id:
            return []
        return [
            v["name"]
            for v in self.command_config.get_voice_options(self._current_repo_id)
        ]

    def _load_model(self, repo_id):
        """加载或切换模型"""
        if self.model and self._current_repo_id == repo_id:
            print(f"模型 {repo_id} 已加载，跳过重复加载")
            return

        if self.model:
            del self.model
            del self.en_pipeline
            del self._voice
            torch.cuda.empty_cache()
            self._current_repo_id = None  # 重置当前模型ID

        print(f"正在加载模型：{repo_id}...")
        try:
            self.model = KModel(repo_id=repo_id).to(self.device).eval()
            self.en_pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=False)
            self._current_repo_id = repo_id
            print(f"模型 {repo_id} 加载完成")
        except Exception as e:
            print(f"模型加载失败：{str(e)}")
            self.model = None
            self.en_pipeline = None
            self._current_repo_id = None  # 重置当前模型ID
            raise

    def load_model(self, repo_id: str):
        """显式加载模型"""
        valid_models = self.command_config.get_config()["commands"][".load"]["params"][
            0
        ]["choices"]
        if repo_id not in valid_models:
            raise ValueError(f"无效模型ID：{repo_id}")
        self._set_repo_id(repo_id)  # 通过_set_repo_id确保逻辑一致

    def generate_audio_data(self, text: str) -> np.ndarray:
        """生成音频数据（不保存文件）"""
        if not self.model:
            raise RuntimeError(
                "模型未加载，请先使用 .load 命令加载模型"
            )  # 强制用户显式加载

        paragraphs = [[text.strip()]]
        wavs = []

        for paragraph in paragraphs:
            if self._join_sentences and len(paragraph) > 1:
                paragraph = ["".join(paragraph)]

            for i, sentence in enumerate(paragraph):
                generator = KPipeline(
                    lang_code="z",
                    repo_id=self._current_repo_id,
                    model=self.model,
                    en_callable=self._custom_en_callable,  # 使用自定义的 en_callable 方法
                )

                audio_result = next(
                    generator(
                        sentence, voice=self._voice, speed=self._default_speed_control
                    )
                )
                wav = audio_result.audio

                if i > 0 and self._n_zeros > 0:
                    wav = np.concatenate([np.zeros(self._n_zeros), wav])

                wavs.append(wav)

        audio_data = np.concatenate(wavs) if wavs else np.array([])
        return audio_data

    def play_audio(self, audio_data: np.ndarray):
        """播放音频数据（需配合音频工具实现）"""
        if not audio_data.size:
            print("音频数据为空，无法播放")
            return

        try:
            import sounddevice as sd

            sd.play(audio_data, samplerate=self._sample_rate)
            sd.wait()
        except ImportError:
            print("播放功能需安装 sounddevice 库：pip install sounddevice")

    def _custom_en_callable(self, text: str) -> str:
        """自定义发音转换逻辑"""
        if text == "Kokoro":
            return "kˈOkəɹO"
        elif text == "Sol":
            return "sˈOl"
        # 其他自定义逻辑
        return next(self.en_pipeline(text)).phonemes  # 默认返回原始文本

    def _default_speed_control(self, len_ps: int) -> float:
        """默认速度控制函数"""
        if len_ps <= 83:
            return 1.0
        elif 83 < len_ps < 183:
            return max(0.8 - (len_ps - 83) / 500 * 0.3, 0.5)
        else:
            return 0.5

    def get_current_config(self):
        return {
            "repo_id": self._current_repo_id,
            "voice": self._voice,
            "sample_rate": self._sample_rate,
            "n_zeros": self._n_zeros,
            "join_sentences": self._join_sentences,
            "device": self.device,
        }

    def set_device(self, device_str: str):
        """设备设置（CPU/GPU）"""
        if device_str not in ["cpu", "cuda"]:
            raise ValueError("device必须为cpu或cuda")
        self.device = device_str
        if self.model:
            self.model.to(self.device)
        print(f"设备已切换至: {device_str}")

    def get_available_models_and_voices(self):
        """获取所有可用模型及其对应的声优"""
        models_info = self.command_config.get_config()["commands"][".load"]["params"][
            0
        ]["choices"]
        table_data = []
        for model in models_info:
            voices = self.command_config.get_voice_options(model)
            for voice in voices:
                table_data.append(
                    [model, voice["name"], voice["language"], voice["gender"]]
                )
        return table_data
