"""资源管理 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource.manager import ResourceManager, Resource

class TestRefCount:
    def test_same_resource_not_loaded_twice(self):
        rm = ResourceManager()
        rm.load_resource("r1", "texture")
        rm.load_resource("r1", "texture")
        assert len(rm.resources) == 1

class TestUnloadCheck:
    def test_used_resource_not_unloaded(self):
        rm = ResourceManager()
        res = rm.load_resource("r1", "texture")
        res.ref_count = 1
        result = rm.unload_resource("r1")
        assert result == False

class TestPoolTypes:
    def test_pool_separated_by_type(self):
        rm = ResourceManager()
        rm.object_pools["enemy"] = [{"type": "enemy"}]
        obj = rm.get_from_pool("bullet")
        assert obj.get("type") == "bullet"

class TestCacheLRU:
    def test_cache_evicts_lru(self):
        rm = ResourceManager()
        for i in range(100):
            rm.cache_resource(f"r{i}", Resource(f"r{i}", "tex"))
        assert len(rm.resources) <= 50

class TestAsyncSync:
    def test_async_load_notifies(self):
        rm = ResourceManager()
        results = []
        rm.load_async("r1", lambda r: results.append(r))
        assert len(results) == 1
