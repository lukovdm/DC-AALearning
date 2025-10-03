import random

from aalpy import SUL
from aalpy.base.SUL import CacheSUL
from lsharpsat.DCValue import DCValue
from lsharpsat.logger import get_logger

log = get_logger()


class RandomDCSUL(SUL):
    def __init__(self, sul: SUL, dc_prob: float = 0.1):
        super().__init__()

        if not isinstance(SUL, CacheSUL):
            sul = CacheSUL(sul)
        self.sul = sul

        self.dc_prob = dc_prob
        self.dc_cache = dict()
        self.path = []

    def pre(self):
        self.sul.pre()
        self.path = []

    def step(self, letter=None):
        out = self.sul.step(letter)

        self.path.append(letter)
        path_tuple = tuple(self.path)
        if path_tuple not in self.dc_cache:
            self.dc_cache[path_tuple] = random.random() < self.dc_prob
            log.debug(f"Setting DC for {path_tuple} to {self.dc_cache[path_tuple]}")

        if self.dc_cache[path_tuple]:
            return DCValue(None)
        else:
            return DCValue(out)

    def post(self):
        self.sul.post()


class OutputDCSUL(SUL):
    def __init__(self, sul: SUL, dc_output, output_map: dict):
        super().__init__()

        if not isinstance(SUL, CacheSUL):
            sul = CacheSUL(sul)
        self.sul = sul

        self.dc_output = dc_output
        self.output_map = output_map

        self.path = []

    def pre(self):
        self.sul.pre()
        self.path = []

    def step(self, letter=None):
        self.path.append(letter)
        out = self.sul.step(letter)
        if out == self.dc_output:
            return DCValue(None)
        else:
            return DCValue(self.output_map[out])

    def post(self):
        self.sul.post()
