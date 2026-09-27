"""资源管理测试 - 8 个，覆盖引用计数/取消/优先级/内存池/拓扑/卸载竞态。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from resources.manager import ResourceManager, ResourceState


class TestRefCount:
    def test_shared_ptr_unloads_only_on_last_release(self):
        rm = ResourceManager()
        h1 = rm.acquire("tex1")
        h2 = rm.acquire("tex1")
        h1.release()
        assert "tex1" in rm.resources, "仍有持有者时不应卸载"
        h2.release()
        assert "tex1" not in rm.resources, "最后一个持有者释放后才卸载"

    def test_release_checks_active_refs(self):
        rm = ResourceManager()
        h1 = rm.acquire("tex1")
        h2 = rm.acquire("tex1")
        h1.release()
        assert "tex1" in rm.resources, "卸载前必须检查活跃引用"
        h2.release()


class TestAsyncCancel:
    def test_cancelled_request_not_executed(self):
        rm = ResourceManager()
        results = []
        req = rm.load_async("tex1", lambda r, d: results.append(r))
        rm.cancel_request(req)
        rm.process_load_queue()
        assert results == []
        assert "tex1" not in rm.resources or rm.resources["tex1"].state != ResourceState.READY

    def test_scene_switch_discards_old_scene_callbacks(self):
        rm = ResourceManager()
        old_scene = object()
        new_scene = object()
        results = []
        rm.load_async("old/tex", lambda r, d: results.append(("old", r)), owner=old_scene)
        rm.load_async("new/tex", lambda r, d: results.append(("new", r)), owner=new_scene)
        assert rm.cancel_by_owner(old_scene) == 1
        rm.process_load_queue()
        assert results == [("new", "new/tex")], "旧场景已取消的回调不应执行"


class TestLoadPriority:
    def test_priority_queue_preempts_and_fifo_tie(self):
        # 高优先级插队：过场小资源排在大贴图前面。
        rm = ResourceManager()
        order = []
        rm.load_async("big_texture", lambda r, d: order.append(r), priority=0)
        rm.load_async("cutscene_tiny", lambda r, d: order.append(r), priority=10)
        rm.process_load_queue()
        assert order == ["cutscene_tiny", "big_texture"], "高优先级必须插队"

        # 同优先级保持 FIFO。
        rm2 = ResourceManager()
        order2 = []
        for rid in ("a", "b", "c"):
            rm2.load_async(rid, lambda r, d: order2.append(r), priority=5)
        rm2.process_load_queue()
        assert order2 == ["a", "b", "c"], "同优先级必须 FIFO"


class TestMemoryPool:
    def test_size_classes_and_overflow_detection(self):
        rm = ResourceManager()
        small = rm.alloc_pool(10)
        big = rm.alloc_pool(1024)
        assert len(small) >= 10
        assert len(big) >= 1024
        small[9] = 1  # 申请范围内允许写
        with pytest.raises(IndexError):
            small[10] = 2  # 超出申请大小必须被拦截
        with pytest.raises(ValueError):
            rm.alloc_pool(10 ** 9)  # 超过最大分级直接拒绝
        rm.free_pool(small)
        with pytest.raises(MemoryError):
            bad = rm.alloc_pool(8)
            bad._block[bad.block_size] = 0  # 直接踩坏金丝雀模拟越界写
            rm.free_pool(bad)


class TestDependencyOrder:
    def test_dependencies_loaded_first_in_topo_order(self):
        rm = ResourceManager()
        rm.declare_dependency("scene", ["mesh", "texture"])
        rm.declare_dependency("mesh", ["material"])
        handle = rm.load_with_deps("scene")
        assert rm.load_order.index("material") < rm.load_order.index("mesh")
        assert rm.load_order.index("mesh") < rm.load_order.index("scene")
        assert rm.load_order.index("texture") < rm.load_order.index("scene")
        ready_events = []
        rm2 = ResourceManager()
        rm2.declare_dependency("scene", ["mesh"])
        rm2.on_ready("mesh", lambda res: ready_events.append("mesh"))
        rm2.on_ready("scene", lambda res: ready_events.append("scene"))
        scene_handle = rm2.load_with_deps("scene")
        assert ready_events == ["mesh", "scene"], "依赖就绪后必须通知依赖者"
        scene_handle.release()
        handle.release()


class TestUnloadRace:
    def test_unload_then_load_completes_is_dropped(self):
        rm = ResourceManager()
        results = []
        req = rm.load_async("tex1", lambda r, d: results.append(r))
        rm.unload("tex1")  # 标记卸载中
        assert rm.resources["tex1"].state == ResourceState.UNLOADING
        rm.process_load_queue()
        assert results == [], "卸载后完成的回调不得重新激活资源"
        assert "tex1" not in rm.resources
        assert req.cancelled is False
