
class FFTBase(object):

    def __init__(self):
        pass

    def analyze(self, f):
        raise NotImplementedError('FFTBase.analyze should be implemented in subclass')

    def analyze_t(self, f):
        raise NotImplementedError('FFTBase.analyze_t should be implemented in subclass')

    def synthesize(self, f_hat):
        raise NotImplementedError('FFTBase.synthesize should be implemented in subclass')

    def synthesize_t(self, f_hat):
        raise NotImplementedError('FFTBase.synthesize_t should be implemented in subclass')
