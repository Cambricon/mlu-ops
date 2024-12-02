'''
    Abstract base class for parsers
'''

__all__ = (
    "_Base_Parser",
)

from abc import ABC, abstractmethod

class _Base_Parser(ABC):
    def __init__(self, parser):
        self.m_parser = parser

    @abstractmethod
    def add_args(self):
        pass

    @abstractmethod
    def add_subparsers(self):
        pass

    def parse_args(self):
        self.add_args()
        self.add_subparsers()
        return self.m_parser.parse_args()
