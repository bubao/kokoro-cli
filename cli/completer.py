# cli/completer.py
import click
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit import PromptSession
from pygments.lexer import RegexLexer
from pygments.token import Literal, Generic
from config.command_config import CommandConfig


class CLICompleter(Completer):
    def __init__(self, command_config=CommandConfig):
        super().__init__()
        self.command_config = command_config
        self.config = self.command_config.get_config()
        self.voice_config = self.command_config.voice_config
        self.param_values = self.config["param_values"]
        self.commands = self.config["commands"]  # 确保 commands 属性存在
        self.voice_options = []  # 初始化为空列表
        self.current_model = None

        # 初始化参数描述字典
        set_params = self.config.get(".set", {}).get("params", [])
        if set_params:
            param_list = set_params[0].get("parameters", [])
            if isinstance(param_list, list):  # 如果参数是列表格式
                self.config["commands"][".set"]["params"][0]["parameters"] = {
                    param["name"]: param["description"] for param in param_list
                }

    def _reload_config(self):
        """重新加载动态配置"""
        click.echo("重新加载配置...")
        self.command_config.reload()  # 调用 reload 更新配置
        self.config = self.command_config.get_config()
        self.voice_config = self.command_config.voice_config
        self.param_values = self.config["param_values"]
        self.commands = self.config["commands"]
        # 确保不覆盖已更新的 voice_options
        if not self.voice_options and self.current_model:
            self._update_voice_completions()

    def set_current_model(self, repo_id: str):
        """设置当前模型并更新补全选项"""
        self.current_model = repo_id
        # 刷新配置以获取最新模型信息
        self._reload_config()  # 调用 _reload_config 更新配置
        self._update_voice_completions()  # 更新 voice_options

    def _update_voice_completions(self):
        """根据当前模型更新 voice 参数补全"""
        if self.current_model:
            self.voice_options = self.command_config.get_param_values(
                "voice", self.current_model
            )
        else:
            self.voice_options = []
            click.echo("当前模型未设置，无法更新语音选项")

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        tokens = [t for t in text.split() if t.strip()]

        if not tokens:
            yield from self._complete_root_commands(text)
            return

        cmd = tokens[0]
        if cmd not in self.commands:  # 确保 commands 属性存在
            yield from self._complete_unknown_command(tokens)
            return

        if cmd == ".set":
            yield from self._complete_set_command(tokens, text)
        elif cmd == ".load":
            yield from self._complete_load_command(tokens, text)

    def _complete_root_commands(self, text):
        """补全根命令"""
        for cmd in self.commands:
            if cmd.startswith(text):
                meta = self.commands[cmd].get("description", "")
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=meta,
                    style="class:command",
                )

    def _complete_unknown_command(self, tokens):
        first_token = tokens[0]
        for cmd in self.commands:
            if cmd.startswith(first_token):
                yield Completion(
                    cmd,
                    start_position=-len(first_token),
                    display_meta=self.commands[cmd].get("description", ""),
                    style="class:command",
                )

    def _complete_load_command(self, tokens, text):
        """补全 .load 命令的模型仓库 ID"""
        if len(tokens) == 1 or text.endswith(" ") == True:
            for repo in self.commands[".load"]["params"][0]["choices"]:
                yield Completion(repo, 0, display_meta="模型仓库")
        elif len(tokens) == 2 or (len(tokens) == 1 and text.endswith(" ") == False):
            current = tokens[1]
            valid_repos = self.commands[".load"]["params"][0]["choices"]
            for repo in valid_repos:
                if repo.startswith(current):
                    yield Completion(repo, -len(current), display_meta="模型仓库")

    def _complete_set_command(self, tokens, text: str):
        """补全 .set 命令的参数和值"""
        if len(tokens) == 1:
            # 补全参数名称
            param_choices = sorted(self.commands[".set"]["params"][0]["choices"])
            for param in param_choices:
                yield Completion(
                    param,
                    start_position=0,
                    display_meta=self._get_param_desc(param),
                    style="class:param",
                )
        elif len(tokens) == 2 or (len(tokens) == 1 and text.endswith(" ") == False):
            # 补全参数值
            param = tokens[1]
            current_value = ""  # 默认为空字符串
            valid_values = self._get_valid_values(param)
            for value in valid_values:
                if value.startswith(current_value):
                    yield Completion(
                        value,
                        start_position=-len(current_value),
                        display_meta=f"参数: {param}",
                        style="class:value",
                    )
        elif len(tokens) == 3 or (len(tokens) == 2 and text.endswith(" ") == False):
            # 补全参数值
            param = tokens[1]
            current_value = tokens[2]
            valid_values = self._get_valid_values(param)  # 统一获取值的方法

            # 确保 param 在 valid_values 里
            if valid_values:
                for value in valid_values:
                    if value.startswith(current_value):
                        yield Completion(
                            value,
                            start_position=-len(current_value),
                            display_meta=f"{param} 参数值",
                            style="class:value",
                        )

    def _get_param_desc(self, param_name):
        """获取参数描述"""
        return self.config["commands"][".set"]["params"][0]["parameters"].get(
            param_name, ""
        )

    def _get_valid_values(self, param):
        """获取有效值"""
        if param == "voice":
            return self.voice_options
        else:
            values = self.param_values.get(param, [])
            return values


# 定义颜色样式
style = Style.from_dict(
    {
        "completion-menu.completion": "bg:#008888 #ffffff",
        "completion-menu.completion.current": "bg:#00aaaa #000000",
        "scrollbar.background": "bg:#88aaaa",
        "scrollbar.button": "bg:#222222",
        "command": "fg:#ff0066 bold",  # 命令颜色
        "param": "fg:#00ff00 italic",  # 参数颜色
        "value": "fg:#ffaa00",  # 值颜色
        "pygments.generic.strong": "fg:#ff0066 bold",  # 命令高亮
        "pygments.generic.emph": "fg:#00ff00 italic",  # 参数高亮
        "pygments.generic.number": "fg:#ffaa00 italic",  # 参数高亮
        "pygments.generic.string": "fg:#169ad0 italic",  # 参数高亮
    }
)


# 自定义 Lexer 高亮命令、参数和值
class CLILexer(RegexLexer):
    tokens = {
        "root": [
            (r"\.[a-z]+ ", Generic.Strong),  # 命令（如 .set）
            (r" \s+[a-z_/]\s+ ", Generic.Emph),  # 参数（如 voice）
            (r"\.*\s+(true|false)", Literal.Number),  # 布尔值
            (r"\.*\s+\d+", Literal.Number),  # 数字参数
            (r"\s+_[a-zA-Z0-9]+\d+", Literal.Number),  # 数字参数
            (r"\.*\s+[a-zA-Z0-9_/]+", Literal.String),  # 一般值
        ]
    }


def init_session(completer=CLICompleter):
    """初始化带补全的命令行会话"""
    # completer = CLICompleter()
    return PromptSession(
        completer=completer,
        style=style,
        lexer=PygmentsLexer(CLILexer),
        history=InMemoryHistory(),  # 添加历史记录支持
    )
