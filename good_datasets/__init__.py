import os
import glob
import importlib
import inspect
import ipdb

__all__ = []

package_dir = os.path.dirname(__file__)

for filepath in glob.glob(os.path.join(package_dir, 'good*.py')):

    module_name = os.path.splitext(os.path.basename(filepath))[0]
    module = importlib.import_module(f".{module_name}", __package__)

    for name, obj in inspect.getmembers(module):

        if inspect.isclass(obj) and name.startswith("GOOD") and obj.__module__ == module.__name__:
            globals()[name] = obj
            
            __all__.append(name)