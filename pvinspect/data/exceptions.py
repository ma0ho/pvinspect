class UnsupportedModalityException(BaseException):
    """This exception indicates that the given imaging modality is not supported"""

    def __init__(self, name):
        super().__init__('Modality "{}" is not supported'.format(name))
