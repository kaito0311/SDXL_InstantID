import inspect 


class A: 
    def __init__(self) -> None:
        self.a = "Hello"
        self.b = "Ka"
        pass


obj = A()

print(
    inspect.signature(obj.__init__).parameters
)