# Kokoro CLI

Kokoro CLI 是一个交互式命令行工具，用于生成和管理文本到语音（TTS）模型。

## 安装

1. 克隆仓库：
    ```sh
    git clone https://github.com/yourusername/kokoro-cli.git
    cd kokoro-cli
    ```

2. 安装依赖：
    ```sh
    pip install -r requirements.txt
	pip install kokoro misaki[zh] misaki[en]
    ```

## 使用

启动交互模式：

```sh
./kc
```

在交互模式中，输入文本直接生成音频，以 `.` 开头为命令。例如：

- `.help` 显示帮助信息
- `.load <repo_id>` 加载/切换模型
- `.set <param> <value>` 设置参数
- `.save <path>` 保存最近生成的音频

## 命令

- `exit` 退出程序
- `help` 显示帮助信息
- `load <repo_id>` 加载/切换模型
- `update` 更新模型缓存信息
- `show` 显示当前配置和模型状态
- `set <param> <value>` 设置参数
- `save <path>` 保存最近生成的音频

## 贡献

欢迎贡献代码！请提交 Pull Request 或报告问题。

## 许可证

本项目采用 MIT 许可证。
