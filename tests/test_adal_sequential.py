import adalflow as adal
from adalflow.core.component import Component
import asyncio

class A(Component):
  def __init__(self, text: str):
    super().__init__()
    self.text = text

  def call(self, a):
    return self.text

  async def acall(self, a):
    return self.text
  
  def __call__(self, *args, **kwargs):
    # Return an awaitable for async context
    return AsyncWrapper(self, args, kwargs)
  
class AsyncWrapper:
  def __init__(self, parent, args, kwargs):
    self.parent = parent
    self.args = args
    self.kwargs = kwargs
  
  def __await__(self):
    async def _process():
      return await self.parent.acall(*self.args, **self.kwargs)
    return _process().__await__()

class B(Component):

  def __init__(self, x: int):
    super().__init__()
    self.x = x

  def call(self, a):
    return self.x
  
  async def acall(self, a):
    return self.x
  
  def __call__(self, *args, **kwargs):
    # Return an awaitable for async context
    return AsyncWrapper(self, args, kwargs)
  
class AsyncWrapper:
  def __init__(self, parent, args, kwargs):
    self.parent = parent
    self.args = args
    self.kwargs = kwargs
  
  def __await__(self):
    async def _process():
      return await self.parent.acall(*self.args, **self.kwargs)
    return _process().__await__()

def main():
  seq = adal.Sequential(A("a"), B(1))
  result = asyncio.run(seq.acall(1))
  print(result)

if __name__ == "__main__":
  main()