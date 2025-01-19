from typing import Any


class SxError(Exception):
    en: str

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def get_code(self) -> str:
        up_cls: str = (
            self.__bases__[0].__name__ if hasattr(self, '__bases__') else ''
        )
        my_cls: str = self.__class__.__name__
        return up_cls + '.' + my_cls

    def __repr__(self) -> str:
        code = self.get_code()
        line = f'[{code}] {self.en}'
        lines = [line]
        for key, val in sorted(self.kwargs.items()):
            line = f'  * {key}: {val}'
            lines.append(line)
        return '\n'.join(lines)
