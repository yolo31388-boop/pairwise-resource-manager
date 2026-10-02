"""资源管理 - 修复版

修复内容:
1. load_resource: 引用计数管理 + 同一资源去重加载
2. unload_resource: 卸载前检查引用计数 + 释放 GPU 资源
3. 对象池: 按类型分离 + 容量上限
4. 资源缓存: LRU 淘汰 + 有效性检查 + 按资源类型区分
5. 异步加载: 线程同步 + 完成通知 + 失败重试
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class Resource:
    rid: str
    rtype: str
    ref_count: int = 0
    data: object = None
    gpu_handle: object = None
    gpu_released: bool = False
    version: int = 0
    path: str | None = None
    mtime: float | None = None


class ResourceManager:
    def __init__(self, cache_capacity: int = 50, pool_max_size: int = 64,
                 max_retries: int = 3):
        self.resources: dict[str, Resource] = {}
        self.object_pools: dict[str, list] = {}
        self.pool_max_size = pool_max_size
        self.cache_capacity = cache_capacity
        self.max_retries = max_retries
        # 缓存按 (rtype, rid) 区分资源类型, OrderedDict 维护 LRU 顺序
        self.cache: OrderedDict[tuple[str, str], Resource] = OrderedDict()
        self._versions: dict[str, int] = {}
        self._lock = threading.RLock()
        self._inflight: dict[str, tuple[threading.Event, dict]] = {}
        self.load_errors: dict[str, Exception] = {}
        self.released_gpu: list = []
        self.completed_loads: list[str] = []
        self._listeners: list = []
        self._gpu_counter = 0
        self.load_count = 0  # 实际加载次数(用于验证去重)

    # ---- bug1: 引用计数 + 去重加载 ----

    def load_resource(self, rid: str, rtype: str, data=None) -> Resource:
        with self._lock:
            existing = self.resources.get(rid)
            if existing is not None:
                return existing
            res = Resource(rid, rtype, data=data)
            res.gpu_handle = self._alloc_gpu_handle(res)
            self.resources[rid] = res
            self.load_count += 1
            return res

    def acquire(self, rid: str, rtype: str = "texture", data=None) -> Resource:
        """获取资源并增加引用计数。"""
        with self._lock:
            res = self.load_resource(rid, rtype, data)
            res.ref_count += 1
            return res

    def release(self, rid: str) -> bool:
        """减少引用计数, 归零时自动卸载。"""
        with self._lock:
            res = self.resources.get(rid)
            if res is None:
                return False
            res.ref_count = max(0, res.ref_count - 1)
            if res.ref_count == 0:
                self.unload_resource(rid)
            return True

    # ---- bug2: 卸载检查引用 + 释放 GPU 资源 ----

    def unload_resource(self, rid: str) -> bool:
        with self._lock:
            res = self.resources.get(rid)
            if res is None:
                return False
            if res.ref_count > 0:
                return False  # 仍在使用, 拒绝卸载
            self._release_gpu(res)
            del self.resources[rid]
            self.cache.pop((res.rtype, rid), None)
            return True

    def _alloc_gpu_handle(self, res: Resource):
        self._gpu_counter += 1
        return f"gpu:{res.rid}#{self._gpu_counter}"

    def _release_gpu(self, res: Resource) -> None:
        if res.gpu_handle is not None:
            self.released_gpu.append(res.gpu_handle)
        res.gpu_handle = None
        res.gpu_released = True

    # ---- bug3: 对象池按类型分离 + 容量上限 ----

    def get_from_pool(self, obj_type: str) -> object:
        with self._lock:
            pool = self.object_pools.setdefault(obj_type, [])
            while pool:
                obj = pool.pop()
                obj_type_of = obj.get("type") if isinstance(obj, dict) else None
                if obj_type_of == obj_type:
                    return obj
                # 类型不符的对象归还到它自己的池子, 绝不返回给调用方
                if obj_type_of is not None:
                    wrong_pool = self.object_pools.setdefault(obj_type_of, [])
                    if len(wrong_pool) < self.pool_max_size:
                        wrong_pool.append(obj)
            return {"type": obj_type}

    def return_to_pool(self, obj: object) -> bool:
        """归还对象到对应类型的池子; 池满则丢弃, 防止无限增长。"""
        obj_type = obj.get("type") if isinstance(obj, dict) else None
        if obj_type is None:
            return False
        with self._lock:
            pool = self.object_pools.setdefault(obj_type, [])
            if len(pool) >= self.pool_max_size:
                return False
            pool.append(obj)
            return True

    # ---- bug4: 缓存 LRU 淘汰 + 有效性检查 + 按类型区分 ----

    def cache_resource(self, rid: str, res: Resource, path: str | None = None) -> None:
        with self._lock:
            res.version = self._versions.get(rid, 0)
            if path is not None:
                res.path = path
                res.mtime = os.path.getmtime(path) if os.path.exists(path) else None
            key = (res.rtype, rid)
            self.cache[key] = res
            self.cache.move_to_end(key)
            self.resources[rid] = res
            while len(self.cache) > self.cache_capacity:
                old_key, old_res = self.cache.popitem(last=False)
                if self.resources.get(old_res.rid) is old_res:
                    del self.resources[old_res.rid]
                self._release_gpu(old_res)

    def get_cached(self, rid: str, rtype: str | None = None) -> Resource | None:
        """读取缓存; 失效(版本过期/文件已修改)的缓存会被淘汰并返回 None。"""
        with self._lock:
            keys = [(rtype, rid)] if rtype is not None else [
                key for key in self.cache if key[1] == rid
            ]
            for key in keys:
                res = self.cache.get(key)
                if res is None:
                    continue
                if not self._is_cache_valid(res):
                    self._evict_cache(key)
                    return None
                self.cache.move_to_end(key)
                return res
            return None

    def _is_cache_valid(self, res: Resource) -> bool:
        if res.version != self._versions.get(res.rid, 0):
            return False
        if res.path is not None and os.path.exists(res.path):
            if res.mtime is not None and os.path.getmtime(res.path) != res.mtime:
                return False
        return True

    def _evict_cache(self, key: tuple[str, str]) -> None:
        res = self.cache.pop(key, None)
        if res is not None and self.resources.get(res.rid) is res:
            del self.resources[res.rid]

    def update_resource(self, rid: str) -> None:
        """资源文件更新后调用, 使旧缓存失效。"""
        with self._lock:
            self._versions[rid] = self._versions.get(rid, 0) + 1

    invalidate_resource = update_resource

    # ---- bug5: 异步加载线程同步 + 完成通知 + 失败重试 ----

    def add_load_listener(self, listener) -> None:
        self._listeners.append(listener)

    def load_async(self, rid: str, callback=None, rtype: str = "async",
                   loader=None, max_retries: int | None = None):
        """加载资源(线程安全)。

        同一资源的并发加载会去重: 只有第一个调用者真正加载,
        其余调用者等待加载完成后拿到同一个资源。
        加载失败按 max_retries 重试; 完成后调用 callback 并通知监听者。
        """
        retries = self.max_retries if max_retries is None else max_retries
        with self._lock:
            existing = self.resources.get(rid)
            if existing is not None and rid not in self._inflight:
                if callback:
                    callback(existing)
                return existing
            inflight = self._inflight.get(rid)
            if inflight is None:
                event, holder = threading.Event(), {}
                self._inflight[rid] = (event, holder)
                owner = True
            else:
                event, holder = inflight
                owner = False

        if not owner:
            # 另一个线程正在加载同一资源, 等待它完成
            event.wait()
            res = holder.get("res") or self.resources.get(rid)
            if res is not None and callback:
                callback(res)
            return res

        res, last_error = None, None
        loader = loader or (lambda: self.load_resource(rid, rtype))
        for _attempt in range(max(1, retries)):
            try:
                res = loader()
                break
            except Exception as exc:  # 加载失败重试
                last_error = exc
        with self._lock:
            if res is None:
                self.load_errors[rid] = last_error
            else:
                self.resources[rid] = res
                self.completed_loads.append(rid)
                holder["res"] = res
            self._inflight.pop(rid, None)
            event.set()

        if res is not None:
            if callback:
                callback(res)
            self._notify_loaded(res)
        return res

    def _notify_loaded(self, res: Resource) -> None:
        for listener in list(self._listeners):
            listener(res)

    def poll_completed(self) -> list[str]:
        """主线程轮询已完成的异步加载。"""
        with self._lock:
            done, self.completed_loads = self.completed_loads, []
            return done
