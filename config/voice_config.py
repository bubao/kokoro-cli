# config/voice_config.py

import json
from pathlib import Path
from huggingface_hub import HfApi
from kokoro import KModel, pipeline
from typing import Dict, Any

CACHE_DIR = Path.home() / ".cache/kokoro"
CACHE_FILE = CACHE_DIR / "models_info.json"


class VoiceConfigManager:
    def __init__(self):
        self.api = HfApi()
        self.gender_codes = {"f": "female", "m": "male"}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_models_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取模型信息，优先使用缓存"""
        if not force_refresh and CACHE_FILE.exists():
            return self._load_cache()

        models_info = self._fetch_from_hf()
        self._save_cache(models_info)
        return models_info

    def _fetch_from_hf(self) -> Dict[str, Any]:
        """从HuggingFace获取最新模型信息"""
        models = {}
        for repo_id in KModel.MODEL_NAMES.keys():
            try:
                files = self.api.list_repo_files(repo_id=repo_id)
                voices = self._parse_voice_files(files)
                languages = self._extract_languages(voices)
                models[repo_id] = {
                    "voices": voices,
                    "languages": list(languages),
                    "default_voice": voices[0]["name"] if voices else None,
                }
            except Exception as e:
                print(f"警告：无法获取{repo_id}信息 - {str(e)}")
                models[repo_id] = {"error": str(e)}
        return {"models": models}

    def _parse_voice_files(self, files: list) -> list:
        """解析语音文件列表"""
        voices = []
        for file_path in files:
            if file_path.startswith("voices/") and file_path.endswith(".pt"):
                filename = file_path.split("/")[-1].split(".")[0]
                if len(filename) < 2:
                    continue  # 跳过无效文件名

                lang_char = filename[0].lower()
                gender_char = filename[1].lower()

                # 解析语言
                lang_code = next(
                    (k for k, v in pipeline.ALIASES.items() if v == lang_char), None
                )
                language = pipeline.LANG_CODES.get(lang_char, "unknown")
                language = lang_code or language.split()[-1].lower()

                # 解析性别
                gender = self.gender_codes.get(gender_char, "unknown")

                voices.append(
                    {"name": filename, "language": language, "gender": gender}
                )
        return voices

    def _extract_languages(self, voices: list) -> set:
        """提取所有语言代码"""
        return {v["language"] for v in voices}

    def _save_cache(self, data: dict):
        """保存缓存文件"""
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_cache(self) -> dict:
        """加载缓存文件"""
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_cache(self):
        """强制更新缓存"""
        self.get_models_info(force_refresh=True)
        print("模型配置已更新")
