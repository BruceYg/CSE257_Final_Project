class HParams:
    '''
    Simple class for hyper-parameter configuration
    '''
    def __init__(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
