"""资源管理 - 15 个测试覆盖 5 类 bug"""
import sys, os, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource.manager import ResourceManager, Resource


class TestRefCount:
    def test_same_resource_not_loaded_twice(self):
        rm = ResourceManager()
        rm.load_resource("r1", "texture")
        rm.load_resource("r1", "texture")
        assert len(rm.resources) == 1
        assert rm.load_count == 1

    def test_acquire_increments_ref_count(self):
        rm = ResourceManager()
        rm.acquire("r1", "texture")
        res = rm.acquire("r1", "texture")
        assert res.ref_count == 2
        assert len(rm.resources) == 1

    def test_release_unloads_when_refcount_zero(self):
        rm = ResourceManager()
        rm.acquire("r1", "texture")
        rm.release("r1")
        assert "r1" not in rm.resources


class TestUnloadCheck:
    def test_used_resource_not_unloaded(self):
        rm = ResourceManager()
        res = rm.load_resource("r1", "texture")
        res.ref_count = 1
        result = rm.unload_resource("r1")
        assert result is False
        assert "r1" in rm.resources

    def test_unused_resource_unloaded(self):
        rm = ResourceManager()
        rm.load_resource("r1", "texture")
        assert rm.unload_resource("r1") is True
        assert "r1" not in rm.resources

    def test_unload_releases_gpu_resource(self):
        rm = ResourceManager()
        res = rm.load_resource("r1", "texture")
        assert res.gpu_handle is not None
        rm.unload_resource("r1")
        assert res.gpu_handle is None
        assert res.gpu_released is True


class TestPoolTypes:
    def test_pool_separated_by_type(self):
        rm = ResourceManager()
        rm.object_pools["enemy"] = [{"type": "enemy"}]
        obj = rm.get_from_pool("bullet")
        assert obj.get("type") == "bullet"
        assert rm.object_pools["enemy"] == [{"type": "enemy"}]

    def test_pool_reuses_same_type(self):
        rm = ResourceManager()
        rm.return_to_pool({"type": "enemy", "id": 1})
        obj = rm.get_from_pool("enemy")
        assert obj["type"] == "enemy"
        assert obj["id"] == 1

    def test_pool_has_size_limit(self):
        rm = ResourceManager(pool_max_size=10)
        for _ in range(100):
            rm.return_to_pool({"type": "bullet"})
        assert len(rm.object_pools["bullet"]) <= 10


class TestCacheLRU:
    def test_cache_evicts_lru(self):
        rm = ResourceManager()
        for i in range(100):
            rm.cache_resource(f"r{i}", Resource(f"r{i}", "tex"))
        assert len(rm.resources) <= 50
        assert len(rm.cache) <= 50

    def test_cache_evicts_least_recently_used(self):
        rm = ResourceManager(cache_capacity=3)
        for i in range(3):
            rm.cache_resource(f"r{i}", Resource(f"r{i}", "tex"))
        rm.get_cached("r0")  # r0 变为最近使用
        rm.cache_resource("r3", Resource("r3", "tex"))
        assert rm.get_cached("r0") is not None
        assert rm.get_cached("r1") is None

    def test_cache_invalidated_on_resource_update(self):
        rm = ResourceManager()
        rm.cache_resource("r1", Resource("r1", "texture"))
        assert rm.get_cached("r1") is not None
        rm.update_resource("r1")  # 资源文件更新
        assert rm.get_cached("r1") is None


class TestAsyncSync:
    def test_async_load_notifies(self):
        rm = ResourceManager()
        results = []
        rm.load_async("r1", lambda r: results.append(r))
        assert len(results) == 1
        assert results[0].rid == "r1"

    def test_concurrent_async_loads_same_resource_once(self):
        rm = ResourceManager()
        calls = []

        def loader():
            calls.append(1)
            time.sleep(0.05)
            return Resource("r1", "async")

        threads = [
            threading.Thread(target=rm.load_async, args=("r1",),
                             kwargs={"loader": loader})
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1
        assert len(rm.resources) == 1

    def test_async_load_retries_after_failure(self):
        rm = ResourceManager()
        attempts = []

        def flaky_loader():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("network error")
            return Resource("rnet", "async")

        res = rm.load_async("rnet", loader=flaky_loader)
        assert res is not None
        assert res.rid == "rnet"
        assert len(attempts) == 3
