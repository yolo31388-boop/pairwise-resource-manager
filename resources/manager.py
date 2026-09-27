"""资源管理：引用计数 / 异步加载 / 优先级队列 / 分级内存池 / 依赖拓扑。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
import itertools
from typing import Callable, Iterable


class ResourceState(Enum):
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"


@dataclass
class Resource:
    rid: str
    data: bytes = b""
    state: ResourceState = ResourceState.LOADING
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    dep_waiting: set[str] = field(default_factory=set)
    ready_listeners: list[Callable[["Resource"], None]] = field(default_factory=list)
    ref_count: int = 0
    active_refs: int = 0
    pending_loads: int = 0


class ResourceHandle:
    """共享指针模式：每个持有者拿到一个句柄，释放时递减共享引用计数。"""

    def __init__(self, manager: "ResourceManager", rid: str, owns_ref: bool = True):
        object.__setattr__(self, "_manager", manager)
        object.__setattr__(self, "_rid", rid)
        object.__setattr__(self, "_released", False)
        object.__setattr__(self, "_owns_ref", owns_ref)
        res = manager.resources.get(rid)
        if res is not None:
            res.active_refs += 1
            if owns_ref:
                res.ref_count += 1

    @property
    def rid(self) -> str:
        return object.__getattribute__(self, "_rid")

    @property
    def resource(self) -> Resource | None:
        return object.__getattribute__(self, "_manager").resources.get(
            object.__getattribute__(self, "_rid")
        )

    @property
    def data(self) -> bytes | None:
        res = self.resource
        return None if res is None else res.data

    def __getattr__(self, name):
        res = object.__getattribute__(self, "_manager").resources.get(
            object.__getattribute__(self, "_rid")
        )
        if res is None:
            raise ValueError(f"资源 {object.__getattribute__(self, '_rid')} 已卸载")
        return getattr(res, name)

    def release(self) -> None:
        if object.__getattribute__(self, "_released"):
            return
        object.__setattr__(self, "_released", True)
        object.__getattribute__(self, "_manager")._release_handle(
            object.__getattribute__(self, "_rid"),
            object.__getattribute__(self, "_owns_ref"),
            explicit=True,
        )

    close = release

    def __enter__(self) -> "ResourceHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self):
        if object.__getattribute__(self, "_released"):
            return
        try:
            manager = object.__getattribute__(self, "_manager")
            rid = object.__getattribute__(self, "_rid")
            owns_ref = object.__getattribute__(self, "_owns_ref")
            object.__setattr__(self, "_released", True)
            manager._release_handle(rid, owns_ref, explicit=False)
        except Exception:
            pass


@dataclass
class LoadRequest:
    token: int
    rid: str
    callback: Callable[[str, bytes], None]
    priority: int
    seq: int
    owner: object | None = None
    cancelled: bool = False


class _PoolBlock(bytearray):
    """带越界检测金丝雀的内存块，容量按大小分级向上取整。"""

    _GUARD = b"\xde\xad\xbe\xef"

    def __init__(self) -> None:
        # 不接收参数：bytearray 的默认 __init__ 会把内容重置成初始长度。
        self.capacity = 0

    @classmethod
    def create(cls, capacity: int) -> "_PoolBlock":
        block = cls()
        super(_PoolBlock, block).__init__(capacity + len(cls._GUARD))
        block.capacity = capacity
        block[capacity:] = cls._GUARD
        return block


class PoolView:
    """只暴露 requested_size 可写区域的内存视图，越界立即抛 IndexError。"""

    __slots__ = ("_mv", "_block", "requested_size", "block_size")

    def __init__(self, block: _PoolBlock, requested_size: int):
        self._block = block
        self._mv = memoryview(block)[:block.capacity]
        self.requested_size = requested_size
        self.block_size = block.capacity

    def __len__(self) -> int:
        return self.requested_size

    def _check_int(self, index: int) -> int:
        if index < 0:
            index += self.requested_size
        if not 0 <= index < self.requested_size:
            raise IndexError("内存块越界写")
        return index

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._mv[: self.requested_size][index]
        return self._mv[self._check_int(index)]

    def __setitem__(self, index, value) -> None:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.requested_size)
            if step == 1:
                region = stop - start
                try:
                    value_len = len(value)
                except TypeError:
                    raise TypeError("切片赋值需要可迭代对象")
                if value_len != region:
                    raise IndexError("内存块越界写：切片超出申请大小")
                self._mv[start:stop] = value
            else:
                indices = list(range(start, stop, step))
                if len(value) != len(indices):
                    raise ValueError("步长切片赋值长度必须匹配")
                for i, v in zip(indices, value):
                    self._mv[i] = v
        else:
            self._mv[self._check_int(index)] = value

    def __iter__(self):
        return iter(self._mv[: self.requested_size])

    def __bytes__(self) -> bytes:
        return bytes(self._mv[: self.requested_size])

    def release(self) -> None:
        raise RuntimeError("请通过 ResourceManager.free_pool 归还内存块")


# 固定大小分级：8, 16, 32, ..., 4096；超出最大级别拒绝分配。
SIZE_CLASSES = tuple(1 << shift for shift in range(3, 13))


class ResourceManager:
    def __init__(
        self,
        size_classes: Iterable[int] = SIZE_CLASSES,
        loader: Callable[[str], bytes] | None = None,
    ):
        self.resources: dict[str, Resource] = {}
        self.load_queue: list[tuple[int, int, LoadRequest]] = []
        self.memory_pools: dict[int, list[bytearray]] = {}
        self.dependencies: dict[str, list[str]] = {}
        self.load_order: list[str] = []
        self._size_classes = tuple(sorted(size_classes))
        self._loader = loader or (lambda rid: b"data:" + rid.encode())
        self._tokens = itertools.count(1)

    # ---- 引用计数（共享指针） ----

    def acquire(self, rid: str) -> ResourceHandle:
        res = self.resources.get(rid)
        if res is None:
            res = Resource(rid=rid)
            self.resources[rid] = res
        elif res.state == ResourceState.UNLOADING:
            # 卸载尚未完成（如等待异步加载）时被重新持有：撤销卸载。
            res.state = ResourceState.LOADING if res.pending_loads else ResourceState.READY
        return ResourceHandle(self, rid, owns_ref=True)

    def release(self, rid: str) -> None:
        self._release_rid(rid)

    def _release_handle(self, rid: str, owns_ref: bool, explicit: bool) -> None:
        res = self.resources.get(rid)
        if res is None:
            return
        if res.active_refs > 0:
            res.active_refs -= 1
        if owns_ref:
            if res.ref_count > 0:
                res.ref_count -= 1
        elif explicit:
            # 非持有句柄（如 load_with_deps 的返回值）：析构不释放资源，
            # 只有显式 release() 才把管理器保留的引用还掉并级联释放依赖。
            if res.ref_count > 0:
                res.ref_count -= 1
        self._try_unload(res)

    def _release_rid(self, rid: str) -> None:
        res = self.resources.get(rid)
        if res is None:
            return
        if res.ref_count > 0:
            res.ref_count -= 1
        self._try_unload(res)

    def _try_unload(self, res: Resource) -> None:
        if (
            res.state != ResourceState.UNLOADING
            and res.ref_count == 0
            and res.active_refs == 0
            and res.pending_loads == 0
        ):
            self._remove(res.rid)

    def unload(self, rid: str) -> None:
        res = self.resources.get(rid)
        if res is None:
            return
        res.state = ResourceState.UNLOADING
        if res.pending_loads > 0:
            # 有正在进行的异步加载：等加载完成后直接释放，不重新激活。
            return
        if res.ref_count == 0 and res.active_refs == 0:
            self._remove(rid)

    def _remove(self, rid: str) -> None:
        res = self.resources.pop(rid, None)
        if res is None:
            return
        # 以管理器依赖表为准（兼容直接写 dependencies 的场景），级联释放依赖。
        dep_rids = list(dict.fromkeys(self.dependencies.get(rid, []) + res.dependencies))
        for dep_rid in dep_rids:
            if dep_rid in self.resources:
                self._release_rid(dep_rid)

    # ---- 异步加载（token 校验 / 取消 / 优先级） ----

    def load_async(
        self,
        rid: str,
        callback: Callable[[str, bytes], None],
        priority: int = 0,
        owner: object | None = None,
    ) -> LoadRequest:
        token = next(self._tokens)
        req = LoadRequest(
            token=token,
            rid=rid,
            callback=callback,
            priority=priority,
            seq=token,
            owner=owner,
        )
        res = self.resources.get(rid)
        if res is None:
            res = Resource(rid=rid)
            self.resources[rid] = res
        res.pending_loads += 1
        # 堆按 (优先级取负, 序号) 排序：高优先级先出队，同级 FIFO，可插队。
        heapq.heappush(self.load_queue, (-priority, req.seq, req))
        return req

    def cancel_request(self, req: LoadRequest) -> None:
        if req is None or req.cancelled:
            return
        req.cancelled = True
        res = self.resources.get(req.rid)
        if res is not None and res.pending_loads > 0:
            res.pending_loads -= 1
            self._try_unload(res)

    def cancel_by_owner(self, owner: object) -> int:
        count = 0
        for _, _, req in list(self.load_queue):
            if not req.cancelled and req.owner is owner:
                self.cancel_request(req)
                count += 1
        return count

    def _is_token_valid(self, req: LoadRequest) -> bool:
        return not req.cancelled

    def process_load_queue(self, max_requests: int | None = None) -> int:
        processed = 0
        while self.load_queue and (max_requests is None or processed < max_requests):
            _, _, req = heapq.heappop(self.load_queue)
            # 回调执行前检查 token：已取消的请求直接丢弃。
            if not self._is_token_valid(req):
                continue
            processed += 1
            try:
                data = self._loader(req.rid)
            except Exception:
                self._drop_request(req)
                continue
            self._complete_request(req, data)
        return processed

    def _drop_request(self, req: LoadRequest) -> None:
        res = self.resources.get(req.rid)
        if res is not None and res.pending_loads > 0:
            res.pending_loads -= 1
            self._try_unload(res)

    def _complete_request(self, req: LoadRequest, data: bytes) -> None:
        res = self.resources.get(req.rid)
        if res is not None and res.pending_loads > 0:
            res.pending_loads -= 1

        if res is not None and res.state == ResourceState.UNLOADING:
            # 资源已在卸载中：加载结果不激活；无新持有者时直接释放。
            if res.ref_count == 0 and res.active_refs == 0:
                self._remove(req.rid)
            return

        if res is None:
            res = Resource(rid=req.rid)
            self.resources[req.rid] = res
        res.data = data
        res.state = ResourceState.READY
        if req.rid not in self.load_order:
            self.load_order.append(req.rid)
        self._notify_ready(res)
        req.callback(req.rid, data)

    # ---- 内存池（按大小分级 + 越界检测） ----

    def _size_class_for(self, size: int) -> int:
        if size <= 0:
            raise ValueError("分配大小必须为正数")
        for block_size in self._size_classes:
            if size <= block_size:
                return block_size
        raise ValueError(
            f"申请大小 {size} 超过最大内存块 {self._size_classes[-1]}"
        )

    def alloc_pool(self, size: int) -> PoolView:
        # 按大小向上取整到分级；不同大小走对应分级的独立池。
        block_size = self._size_class_for(size)
        pool = self.memory_pools.setdefault(block_size, [])
        if not pool:
            pool.extend(_PoolBlock.create(block_size) for _ in range(10))
        return PoolView(pool.pop(), size)

    def free_pool(self, block: PoolView) -> None:
        raw = block._block
        guard = bytes(raw[raw.capacity:])
        if guard != _PoolBlock._GUARD:
            raise MemoryError(f"内存块越界写：金丝雀被破坏 (size={raw.capacity})")
        self.memory_pools.setdefault(raw.capacity, []).append(raw)

    # ---- 依赖（拓扑排序 + 就绪通知） ----

    def declare_dependency(self, rid: str, deps: str | Iterable[str]) -> None:
        if isinstance(deps, str):
            deps = [deps]
        deps = list(deps)
        known = self.dependencies.setdefault(rid, [])
        for dep in deps:
            if dep not in known:
                known.append(dep)
        res = self.resources.get(rid)
        if res is None:
            res = Resource(rid=rid)
            self.resources[rid] = res
        for dep in deps:
            if dep not in res.dependencies:
                res.dependencies.append(dep)
            dep_res = self.resources.get(dep)
            if dep_res is None:
                dep_res = Resource(rid=dep)
                self.resources[dep] = dep_res
            if rid not in dep_res.dependents:
                dep_res.dependents.append(rid)

    def _topo_order(self, rid: str) -> list[str]:
        order: list[str] = []
        visited: set[str] = set()
        stack: set[str] = set()

        def visit(node: str) -> None:
            if node in visited:
                return
            if node in stack:
                raise ValueError(f"资源依赖存在环: {node}")
            stack.add(node)
            for dep in self.dependencies.get(node, []):
                visit(dep)
            stack.discard(node)
            visited.add(node)
            order.append(node)

        visit(rid)
        return order

    def load_with_deps(self, rid: str) -> ResourceHandle:
        order = self._topo_order(rid)
        # 确保每个节点的 Resource 存在且 dependencies 字段同步。
        for node in order:
            res = self.resources.get(node)
            if res is None:
                res = Resource(rid=node)
                self.resources[node] = res
            for dep in self.dependencies.get(node, []):
                if dep not in res.dependencies:
                    res.dependencies.append(dep)
                dep_res = self.resources.setdefault(dep, Resource(rid=dep))
                if node not in dep_res.dependents:
                    dep_res.dependents.append(node)

        # 统计每个资源被多少个已加载资源直接持有（菱形依赖各父节点各持一份）。
        holders: dict[str, int] = {}
        for node in order:
            for dep in self.dependencies.get(node, []):
                holders[dep] = holders.get(dep, 0) + 1

        target: ResourceHandle | None = None
        for node in order:
            res = self.resources[node]
            res.data = self._loader(node)
            res.state = ResourceState.READY
            if node not in self.load_order:
                self.load_order.append(node)
            # 目标资源交给调用者持有；其余资源由依赖它的已加载资源持有，
            # 目标资源卸载时通过 dependencies 级联释放。
            res.ref_count += 1 if node == rid else holders.get(node, 0)
            if node == rid:
                # 返回非持有句柄：丢弃返回值不卸载（资源由管理器保留），
                # 显式 release()/unload() 才释放目标及其依赖。
                target = ResourceHandle(self, node, owns_ref=False)
            self._notify_ready(res)
        return target

    def _notify_ready(self, res: Resource) -> None:
        listeners = res.ready_listeners
        res.ready_listeners = []
        for listener in listeners:
            listener(res)
        for dependent_rid in list(res.dependents):
            dependent = self.resources.get(dependent_rid)
            if dependent is None:
                continue
            dependent.dep_waiting.discard(res.rid)
            if not dependent.dep_waiting and dependent.state == ResourceState.READY:
                self._notify_ready(dependent)

    def on_ready(self, rid: str, listener: Callable[[Resource], None]) -> None:
        res = self.resources.get(rid)
        if res is not None and res.state == ResourceState.READY:
            listener(res)
        elif res is not None:
            res.ready_listeners.append(listener)
