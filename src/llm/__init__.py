"""多模型统一接入层（对应计划 M1）。

对外只暴露一个 ``get_chat_model(provider)`` 风格的入口，
屏蔽 DeepSeek 与小米 MiMo 在鉴权、地址、模型名上的差异。
"""
