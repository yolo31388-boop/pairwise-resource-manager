"""资源管理测试 - 8个失败"""
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine import ResourceManager

class TestRefCount:
    def test_release_checks_active_refs(self):
        rm = ResourceManager()
        r1 = rm.acquire("tex1")
        r2 = rm.acquire("tex1")
        rm.release("tex1")
        assert "tex1" in rm.resources, "还有引用时被卸载"

class TestAsyncCancel:
    def test_cancelled_request_not_executed(self):
        rm = ResourceManager()
        results = []
        rm.load_async("tex1", lambda r, d: results.append(r))
        rm.load_queue.clear()  # 模拟取消
        rm.process_load_queue()
        assert len(results) == 0, "已取消请求仍执行"

class TestPoolOverflow:
    def test_large_alloc_no_overflow(self):
        rm = ResourceManager()
        block = rm.alloc_pool(1024)
        assert len(block) >= 1024, "大对象越界"

class TestDependencyOrder:
    def test_dependencies_loaded_first(self):
        rm = ResourceManager()
        rm.dependencies["scene"] = ["mesh", "texture"]
        rm.load_with_deps("scene")
        assert "mesh" in rm.resources and "texture" in rm.resources
