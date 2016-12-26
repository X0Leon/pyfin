# -*- coding: utf-8 -*-

from . import core
from . import data

from .data import get
from .core import *

core.extend_pandas()

__version__ = '1.0.0'
__author__ = 'Leon Zhang'
