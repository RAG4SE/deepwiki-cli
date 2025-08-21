
class AsyncWrapper:
    """
    Async wrapper class for returning awaitable objects inside functions.
    
    This class allows components to be called in async contexts by wrapping
    synchronous calls and providing an async interface. Mainly used to support
    components in async pipelines like adalflow.Sequential.
    
    Usage example:
        Return this wrapper in a component's __call__ method as long as the component has an acall method:

		def acall(self, *args, **kwargs):
			...

        def __call__(self, *args, **kwargs):
            return self.AsyncWrapper(self, args, kwargs)
        
        Then it can be used in async functions:
        
        async def process():
            result = await component(*args, **kwargs)
            return result
    """
    def __init__(self, parent, args, kwargs):
      self.parent = parent
      self.args = args
      self.kwargs = kwargs
    
    def __await__(self):
      async def _process():
        return await self.parent.acall(*self.args, **self.kwargs)
      return _process().__await__()