class Summary:
    class __Summary:
        def __init__(self):
            self._values = {}
            self._enabled = False

        def enable(self):
            self._enabled = True

        def disable(self):
            self._enabled = False

        def put(self, k, v):
            if self._enabled:
                if k not in self._values.keys():
                    self._values[k] = v
                else:
                    raise RuntimeError("key {} already used".format(k))

        def get(self, k):
            return self._values.get(k)

        def flush(self):
            self._values = {}

        def keys(self):
            return self._values.keys()

    _instances = {}

    def __init__(self, key):
        if Summary._instances.get(key) is None:
            Summary._instances[key] = Summary.__Summary()
        self._key = key

    def __getattr__(self, name):
        return getattr(self._instances[self._key], name)
