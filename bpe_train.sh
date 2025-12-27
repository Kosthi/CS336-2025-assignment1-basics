# 同步依赖（没有 cppyy 了）
uv sync

# 用 uv 的 Python 构建 C++ 模块
uv run cmake -S cpp -B cpp/build
uv run cmake --build cpp/build

uv run pytest tests/test_train_bpe.py
