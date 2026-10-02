"""资源管理。

修复点：
1. load_resource 对同一 rid 做去重与引用计数，同一资源只保留一份。
2. unload_resource 先检查引用计数，仍被引用时拒绝卸载；真正卸载时
   同时释放 CPU 数据与 GPU 资源。
3. 对象池按类型分离，取出时校验对象类型，且每类池有容量上限。
4. 缓存按 (rtype, rid) 区分类型，LRU 淘汰，支持版本/文件修改时间的
   有效性检查，缓存不会命中过期资源。
5. load_async 加锁同步：同一资源只会加载一次，加载完成后通过事件通知
   主线程，未就绪时取不到资源；自定义 loader 失败会自动重试。
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Resource:
    rid: str
    rtype: str
    ref_count: int = 0
    data: object = None
    # GPU 资源句柄（纹理/缓冲对象等），卸载时必须一起释放
    gpu_handle: object = None
    # 资源版本/来源文件，用于缓存有效性检查
    version: object = None
    source_path: str | None = None
    gpu_allocated: bool = False

    def allocate_gpu(self, handle=None) -> None:
        """模拟把资源上传到 GPU。"""
        if not self.gpu_allocated:
            self.gpu_handle = handle if handle is not None else f"gpu:{self.rtype}:{self.rid}"
            self.gpu_allocated = True

    def release_gpu(self) -> bool:
        """释放 GPU 资源，返回是否实际发生了释放。"""
        if not self.gpu_allocated:
            return False
        cleanup = getattr(self.gpu_handle, "release", None)
        if callable(cleanup):
            cleanup()
        self.gpu_handle = None
        self.gpu_allocated = False
        return True


class ResourceLoadError(Exception):
    """异步加载（含重试）最终失败。"""


@dataclass
class _AsyncState:
    event: threading.Event = field(default_factory=threading.Event)
    resource: Resource | None = None
    error: Exception | None = None
    attempts: int = 0


class ResourceManager:
    def __init__(
        self,
        cache_size: int = 50,
        max_pool_size: int = 100,
        max_async_retries: int = 3,
    ):
        self.resources: dict[str, Resource] = {}
        # 对象池按类型分离，每类独立、有上限
        self.object_pools: dict[str, list] = {}
        self.max_pool_size = max_pool_size

        # LRU 缓存：键为 (rtype, rid)，纹理与模型不会混在一起
        self._cache: OrderedDict[tuple[str, str], Resource] = OrderedDict()
        self.cache_size = cache_size

        # rid -> 异步加载状态，保证同一资源只加载一次
        self._async_states: dict[str, _AsyncState] = {}

        self.max_async_retries = max_async_retries
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 1. 加载与引用计数
    # ------------------------------------------------------------------
    def load_resource(
        self,
        rid: str,
        rtype: str,
        data=None,
        version: object = None,
        source_path: str | None = None,
    ) -> Resource:
        """加载资源。同一 rid 只保留一份，重复加载只增加引用计数。"""
        with self._lock:
            res = self.resources.get(rid)
            if res is not None:
                if res.rtype != rtype:
                    raise TypeError(
                        f"资源 {rid} 已按类型 {res.rtype} 加载，不能作为 {rtype} 重复加载"
                    )
                res.ref_count += 1
                return res

            res = Resource(
                rid=rid,
                rtype=rtype,
                ref_count=1,
                data=data,
                version=version,
                source_path=source_path,
            )
            # 重新加载意味着磁盘上的资源已更新，旧缓存一律失效
            self.invalidate_cache(rid)
            self.resources[rid] = res
            return res

    def acquire_resource(self, rid: str) -> Resource | None:
        """获取已加载资源并把引用计数加一。"""
        with self._lock:
            res = self.resources.get(rid)
            if res is None:
                return None
            res.ref_count += 1
            return res

    def release_resource(self, rid: str) -> int:
        """释放一次引用，返回剩余引用数。"""
        with self._lock:
            res = self.resources.get(rid)
            if res is None:
                return 0
            if res.ref_count > 0:
                res.ref_count -= 1
            return res.ref_count

    # ------------------------------------------------------------------
    # 2. 卸载：检查引用 + 释放 GPU
    # ------------------------------------------------------------------
    def unload_resource(self, rid: str, force: bool = False) -> bool:
        """卸载资源。

        ref_count > 0 且未指定 force 时拒绝卸载；真正卸载时先释放 GPU
        资源，再清理 CPU 数据与缓存引用。
        """
        with self._lock:
            res = self.resources.get(rid)
            if res is None:
                return False
            if res.ref_count > 0 and not force:
                return False

            res.release_gpu()
            res.data = None

            cache_key = (res.rtype, rid)
            if cache_key in self._cache:
                del self._cache[cache_key]

            del self.resources[rid]
            return True

    # ------------------------------------------------------------------
    # 3. 对象池：按类型分离、类型校验、容量上限
    # ------------------------------------------------------------------
    def get_from_pool(self, obj_type: str, factory=None) -> object:
        """从指定类型的池中取对象。"""
        with self._lock:
            pool = self.object_pools.setdefault(obj_type, [])
            while pool:
                obj = pool.pop()
                if self._object_type(obj) == obj_type:
                    return obj
                # 丢弃混入的错误类型对象，绝不能返回给调用方
            if factory is not None:
                return factory(obj_type)
            return {"type": obj_type}

    def return_to_pool(self, obj: object) -> bool:
        """把对象归还到对应类型的池，池满则拒绝回收。"""
        obj_type = self._object_type(obj)
        if obj_type is None:
            return False
        with self._lock:
            pool = self.object_pools.setdefault(obj_type, [])
            if len(pool) >= self.max_pool_size:
                return False
            pool.append(obj)
            return True

    @staticmethod
    def _object_type(obj: object) -> str | None:
        if isinstance(obj, dict):
            return obj.get("type")
        return getattr(obj, "type", None)

    # ------------------------------------------------------------------
    # 4. 缓存：LRU 淘汰 + 有效性检查 + 类型区分
    # ------------------------------------------------------------------
    def cache_resource(self, rid: str, res: Resource) -> bool:
        """缓存资源。

        - 按 (rtype, rid) 区分资源类型；
        - 同一 rid 已有更新版本时旧缓存失效；
        - 超过容量时按 LRU 淘汰，且不会淘汰仍被引用的资源。
        """
        if not isinstance(res, Resource):
            raise TypeError("只能缓存 Resource 对象")

        with self._lock:
            current = self.resources.get(rid)
            if current is not None and current is not res:
                if current.rtype == res.rtype:
                    # 同类型：已有更新版本时旧缓存直接失效
                    if self._is_newer(current, res):
                        return False
                    current.release_gpu()
                    current.data = None
                    self.invalidate_cache(rid, rtype=res.rtype)
                # 不同类型共存：缓存按 (rtype, rid) 区分，互不覆盖

            self.resources[rid] = res

            key = (res.rtype, rid)
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                self._cache[key] = res
                self._evict_cache_locked()
            return True

    def get_cached_resource(self, rid: str, rtype: str | None = None) -> Resource | None:
        """读取缓存；过期或类型不符时返回 None 并清理失效项。"""
        with self._lock:
            if rtype is not None:
                key = (rtype, rid)
                res = self._cache.get(key)
                if res is not None and self._is_stale_locked(res):
                    self._cache.pop(key, None)
                    return None
                if res is not None:
                    self._cache.move_to_end(key)
                return res

            # 未指定类型时也不允许跨类型误取
            matched = [
                (key, res)
                for key, res in self._cache.items()
                if key[1] == rid
            ]
            for key, res in matched:
                if self._is_stale_locked(res):
                    self._cache.pop(key, None)
            valid = [(key, res) for key, res in matched if (key, res) in self._cache.items()]
            if not valid:
                return None
            key, res = valid[0]
            self._cache.move_to_end(key)
            return res

    def invalidate_cache(self, rid: str, rtype: str | None = None) -> None:
        with self._lock:
            keys = [
                key
                for key in self._cache
                if key[1] == rid and (rtype is None or key[0] == rtype)
            ]
            for key in keys:
                self._cache.pop(key, None)

    def _evict_cache_locked(self) -> None:
        while len(self._cache) > self.cache_size:
            key, victim = self._cache.popitem(last=False)
            # 正在被使用的资源不能被淘汰，移到队尾稍后再试
            if victim.ref_count > 0:
                self._cache[key] = victim
                self._cache.move_to_end(key)
                if self._all_cache_entries_busy_locked():
                    break
                continue
            if key[1] in self.resources and self.resources[key[1]] is victim:
                victim.release_gpu()
                victim.data = None
                del self.resources[key[1]]

    def _all_cache_entries_busy_locked(self) -> bool:
        return all(res.ref_count > 0 for res in self._cache.values())

    @staticmethod
    def _is_newer(candidate: Resource, incoming: Resource) -> bool:
        """candidate 是否比 incoming 更新。"""
        if incoming.version is not None and candidate.version is not None:
            return candidate.version > incoming.version
        return False

    def _is_stale_locked(self, res: Resource) -> bool:
        if res.source_path:
            try:
                mtime = os.path.getmtime(res.source_path)
            except OSError:
                return False
            if res.version is not None and mtime > res.version:
                return True
        return False

    # ------------------------------------------------------------------
    # 5. 异步加载：去重、同步、通知、重试
    # ------------------------------------------------------------------
    def load_async(
        self,
        rid: str,
        callback=None,
        rtype: str = "async",
        loader=None,
        data=None,
        max_retries: int | None = None,
        timeout: float | None = None,
    ) -> threading.Event:
        """异步加载资源，返回一个在加载完成时 set 的事件。

        - 同一 rid 的并发请求只会触发一次实际加载；
        - 默认 loader 在调用线程内完成（向后兼容），自定义 loader 在
          后台线程执行；
        - 加载失败按 max_retries 自动重试；
        - 回调在加载成功后收到资源，主线程可用事件或 is_ready 感知。
        """
        with self._lock:
            state = self._async_states.get(rid)
            is_owner = False
            if state is None:
                state = _AsyncState()
                self._async_states[rid] = state
                is_owner = True
            elif state.event.is_set() and state.error is not None:
                # 上次加载最终失败，允许重新发起加载
                state = _AsyncState()
                self._async_states[rid] = state
                is_owner = True

        def _notify() -> None:
            if callback is not None and state.error is None:
                callback(state.resource)

        if not is_owner:
            # 已有线程在加载：等待同一结果，杜绝重复加载
            state.event.wait(timeout)
            _notify()
            return state.event

        retries = self.max_async_retries if max_retries is None else max_retries

        def _run() -> None:
            last_error: Exception | None = None
            for attempt in range(retries + 1):
                state.attempts = attempt + 1
                try:
                    loaded_data = loader(rid) if loader is not None else data
                    with self._lock:
                        res = self.load_resource(rid, rtype, data=loaded_data)
                        state.resource = res
                    state.event.set()
                    if callback is not None:
                        callback(res)
                    return
                except Exception as exc:  # 网络/IO 失败：记录后重试
                    last_error = exc
            state.error = last_error or ResourceLoadError(f"加载资源 {rid} 失败")
            with self._lock:
                # 失败后移除，后续 load_async 可以重新发起
                if self._async_states.get(rid) is state:
                    self._async_states.pop(rid, None)
            state.event.set()

        if loader is None:
            # 默认行为保持同步，回调在返回前一定已执行
            _run()
        else:
            thread = threading.Thread(target=_run, name=f"load_async:{rid}", daemon=True)
            thread.start()
            if timeout is not None:
                state.event.wait(timeout)
                _notify()
        return state.event

    def is_ready(self, rid: str) -> bool:
        state = self._async_states.get(rid)
        return state is not None and state.event.is_set() and state.error is None

    def get_async_result(self, rid: str, timeout: float | None = None) -> Resource | None:
        """等待异步加载完成并返回资源；未就绪或失败时返回 None。"""
        state = self._async_states.get(rid)
        if state is None:
            return self.resources.get(rid)
        if not state.event.wait(timeout):
            return None
        return None if state.error is not None else self.resources.get(rid)
