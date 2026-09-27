"""资源管理。

修复后的实现位于 resources.manager；此处保留兼容入口。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from resources.manager import (  # noqa: E402,F401
    Resource,
    ResourceState,
    ResourceHandle,
    LoadRequest,
    ResourceManager,
)
