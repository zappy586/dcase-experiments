"""Tiny name -> class registries so components can be swapped from a YAML config."""


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str):
        def deco(cls):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[name] = cls
            return cls

        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name]

    def names(self):
        return sorted(self._items)


ENCODERS = Registry("encoder")
BOTTLENECKS = Registry("bottleneck")
DECODERS = Registry("decoder")
