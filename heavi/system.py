from abc import ABC
from dataclasses import dataclass
from multiprocessing.sharedctypes import Value
from random import random
from typing import Callable
from rfcircuit import *


class Param:
    value: float = None

    def __post_init__(self):
        if isinstance(self.value, float):
            self.get = lambda: self.value
        elif isinstance(self.value, (list, tuple):
            self.get = lambda: random() * (max(self.value)-min(self.value) + min(self.value)



class SystemPart:
    def __init__(self, factory: Callable, arguments: dict):
        self._factory = factory
        self._args = arguments

    def _compile(self, node1: Node, node2: Node):
        kwargs = {x, self._args[x].get() for x in self._args}
        self._factory(node1, node2, **kwargs)


class System:
    def __init__(self, Z0: complex = 50):
        self.network = Network()
        self._gnd = self.network.gnd
        self._nin = self.network.node("vp1")
        self._nout = self.network.node("vp2")
        self._z0 = 50
        self._p1 = self.network.terminal(self._gnd, self._nin, self._z0)
        self._p2 = self.network.terminal(self._gnd, self._nout, self._z0)
        self._components = list()

    def add_cable(self, length: Param, IL: Param, VSWR: Param):
        G = lambda: (1 - VSWR.get())/(1 + VSWR.get())
        Zcable = lambda: -self._z0*