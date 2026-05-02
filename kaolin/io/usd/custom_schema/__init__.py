from pxr import Plug
import pathlib
plugin = Plug.Registry().RegisterPlugins([(pathlib.Path(__file__).parent).absolute().as_posix()])

__all__ = [
    'plugin'
]
