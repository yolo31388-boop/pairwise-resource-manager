"""资源管理 - 6个bug已修复"""
from dataclasses import dataclass
import heapq
import itertools


@dataclass
class Resource:
    rid: str
    data: bytes
    ref_count: int = 0
    loading: bool = False
    unloading: bool = False


@dataclass
class LoadRequest:
    """异步加载请求token，取消后置 cancelled，回调执行前校验"""
    rid: str
    callback: object
    cancelled: bool = False


class ResourceManager:
    POOL_TIERS = (64, 256, 1024, 4096, 16384, 65536)
    GUARD = b"\xDE\xAD\xBE\xEF"  # 块尾哨兵，用于越界写检测

    def __init__(self):
        self.resources: dict[str, Resource] = {}
        self.load_queue: list = []  # (-priority, seq, LoadRequest) 堆
        self._seq = itertools.count()
        self.pending: dict[str, list[LoadRequest]] = {}
        self.memory_pools: dict[int, list] = {}
        self.dependencies: dict[str, list[str]] = {}
        self._dependents: dict[str, list[str]] = {}
        self._load_listeners: dict[str, list] = {}

    # ---- bug1: 引用计数（共享指针语义，最后一个持有者释放才卸载）----
    def acquire(self, rid: str) -> Resource:
        res = self.resources.get(rid)
        if res is None:
            res = Resource(rid, b"")
            self.resources[rid] = res
        res.ref_count += 1
        return res

    def release(self, rid: str):
        res = self.resources.get(rid)
        if res is None:
            return
        res.ref_count -= 1
        if res.ref_count <= 0:
            res.ref_count = 0
            self._unload(rid, res)

    def _unload(self, rid: str, res: Resource):
        if res.ref_count > 0:
            return  # 卸载前检查活跃引用
        # bug6: 标记卸载中，取消进行中的异步加载，完成回调不得重新激活
        res.unloading = True
        for req in self.pending.get(rid, []):
            req.cancelled = True
        del self.resources[rid]

    # ---- bug2/bug3: 取消token + 优先级队列（高优先级插队）----
    def load_async(self, rid: str, callback, priority: int = 0) -> LoadRequest:
        req = LoadRequest(rid, callback)
        heapq.heappush(self.load_queue, (-priority, next(self._seq), req))
        self.pending.setdefault(rid, []).append(req)
        return req

    def cancel_load(self, req: LoadRequest):
        req.cancelled = True

    def process_load_queue(self):
        while self.load_queue:
            _, _, req = heapq.heappop(self.load_queue)
            if req.cancelled:
                continue  # bug2: 已取消的请求直接丢弃
            pend = self.pending.get(req.rid)
            if pend and req in pend:
                pend.remove(req)
            res = self.resources.get(req.rid)
            if res is not None and res.unloading:
                continue  # bug6: 卸载中的资源不重新激活
            if res is not None:
                res.loading = True
                res.data = b"data"
                res.loading = False
            req.callback(req.rid, b"data")

    # ---- bug4: 按大小分级的内存池 + 越界检测 ----
    def _tier_for(self, size: int) -> int:
        for tier in self.POOL_TIERS:
            if tier >= size:
                return tier
        return size  # 超大对象独占块

    def alloc_pool(self, size: int) -> bytearray:
        tier = self._tier_for(size)
        pool = self.memory_pools.setdefault(tier, [])
        if pool:
            return pool.pop()
        block = bytearray(tier + len(self.GUARD))
        block[-len(self.GUARD):] = self.GUARD  # 尾部哨兵
        return block

    def free_pool(self, block: bytearray):
        if bytes(block[-len(self.GUARD):]) != self.GUARD:
            raise RuntimeError("内存池块被越界写")
        tier = len(block) - len(self.GUARD)
        self.memory_pools.setdefault(tier, []).append(block)

    # ---- bug5: 依赖拓扑排序，被依赖的先加载，加载完通知依赖者 ----
    def load_with_deps(self, rid: str) -> Resource:
        for dep in self._topo_sort(rid):
            self.acquire(dep)
            self._notify_dependents(dep)
        return self.resources[rid]

    def _topo_sort(self, rid: str) -> list[str]:
        order, visited, in_stack = [], set(), set()

        def visit(node):
            if node in visited:
                return
            if node in in_stack:
                raise ValueError(f"循环依赖: {node}")
            in_stack.add(node)
            for dep in self.dependencies.get(node, []):
                self._dependents.setdefault(dep, []).append(node)
                visit(dep)
            in_stack.discard(node)
            visited.add(node)
            order.append(node)

        visit(rid)
        return order

    def on_loaded(self, rid: str, listener):
        self._load_listeners.setdefault(rid, []).append(listener)

    def _notify_dependents(self, rid: str):
        for listener in self._load_listeners.pop(rid, []):
            listener(rid)
