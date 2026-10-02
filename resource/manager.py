"""资源管理 - 含5个bug"""
from dataclasses import dataclass, field

@dataclass
class Resource:
    rid: str
    rtype: str
    ref_count: int = 0
    data: object = None

class ResourceManager:
    def __init__(self):
        self.resources: dict[str, Resource] = {}
        self.object_pools: dict[str, list] = {}  # bug3: 不按类型分离

    def load_resource(self, rid: str, rtype: str, data=None) -> Resource:
        # bug1: 不做引用计数，重复加载
        res = Resource(rid, rtype, data=data)
        self.resources[rid] = res
        return res

    def unload_resource(self, rid: str) -> bool:
        # bug2: 不检查引用，直接卸载
        if rid in self.resources:
            del self.resources[rid]
        return True

    def get_from_pool(self, obj_type: str) -> object:
        # bug3: 不检查类型，无上限
        if obj_type in self.object_pools and self.object_pools[obj_type]:
            return self.object_pools[obj_type].pop()
        return {"type": obj_type}

    def cache_resource(self, rid: str, res: Resource) -> None:
        # bug4: 无LRU，不检查有效性
        self.resources[rid] = res

    def load_async(self, rid: str, callback=None) -> None:
        # bug5: 无线程同步，无通知
        res = self.load_resource(rid, "async")
        if callback:
            callback(res)
