"""资源管理 - 带6个bug"""
from dataclasses import dataclass, field
import heapq

@dataclass
class Resource:
    rid: str
    data: bytes
    ref_count: int = 0
    loading: bool = False

class ResourceManager:
    def __init__(self):
        self.resources: dict[str, Resource] = {}
        self.load_queue: list = []
        self.memory_pools: dict[int, list] = {}
        self.dependencies: dict[str, list[str]] = {}

    def acquire(self, rid: str) -> Resource:
        if rid in self.resources:
            res = self.resources[rid]
            res.ref_count += 1
            return res
        res = Resource(rid, b"")
        self.resources[rid] = res
        res.ref_count = 1
        return res

    def release(self, rid: str):
        if rid in self.resources:
            res = self.resources[rid]
            res.ref_count -= 1
            if res.ref_count <= 0:
                # bug1: 不检查活跃引用直接卸载
                del self.resources[rid]

    def load_async(self, rid: str, callback):
        # bug2: 不检查请求是否取消
        self.load_queue.append((rid, callback))

    def process_load_queue(self):
        while self.load_queue:
            rid, cb = self.load_queue.pop(0)
            cb(rid, b"data")

    def alloc_pool(self, size: int) -> bytearray:
        # bug4: 固定块大小，大对象越界
        if size not in self.memory_pools:
            self.memory_pools[size] = [bytearray(size) for _ in range(10)]
        if self.memory_pools[size]:
            return self.memory_pools[size].pop()
        return bytearray(size)

    def load_with_deps(self, rid: str):
        # bug5: 不按拓扑排序
        deps = self.dependencies.get(rid, [])
        for dep in deps:
            self.acquire(dep)
        return self.acquire(rid)
