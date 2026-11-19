维护的源码、脚本、案例模板和生成的源码禁止新增普通注释。保留编译器、解释器和工具实际使用的指令。

跨应用、跨精度执行相同功能的公共算子只维护一份实现。

提交前运行 `python3 tool/check_source_comments.py`。该检查器的依赖列在 `tool/requirements-source-comments.txt`。
