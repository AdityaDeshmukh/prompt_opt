class BaseScore: 
    def __call__(self, *args, **kwargs): 
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs): 
        raise NotImplementedError