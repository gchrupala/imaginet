# Mock interface for a composable layer


class Layer(object):

    def __init__(self):
        self.params = []

    def apply(self, input):
        raise NotImplementedError

    def compose(self, l2):
        l = Layer()
        l.apply = lambda input: self.apply(l2.apply(input))
        l.params = self.params + l2.params
        return l


class Embedding(Layer):
    
    def __init__(self, size_in, size_out):
        self.size = size
        self.E = init((size_in, size_out))
        self.params = [self.E]

    def apply(self, input):
        return self.E[input]
        

class Dense(Layer):
    
    def __init__(self, size_in, size_out, activation):
        self.size_in = size_in
        self.size_out = size_out
        self.activation = activation
        self.w = init((self.size_in, self.size_out))
        self.b = shared0((self.size_out))

    def apply(self, input):
        return activation(T.dot(input, self.w) + self.b)

# Need shared parameters
# Fork and Merge layers

# layers = [A, B, C]

# model = C.compose(B.compose(A))
# out = model.apply(input)
