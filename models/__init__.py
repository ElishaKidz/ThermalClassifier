import pkgutil

class ModelRepo:
    registry = {}

    @classmethod
    def register(cls, name, item):
        cls.registry[name] = item

# Dynamically import and register model classes
package = __name__
for importer, module_name, _ in pkgutil.iter_modules([__path__[0]]):
    module = importer.find_module(module_name).load_module(module_name)
    for item_name in dir(module):
        item = getattr(module, item_name)
        if callable(item) and hasattr(item, "__module__") and item.__module__ == module_name:
            ModelRepo.register(item_name, item)
