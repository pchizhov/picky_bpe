from collections import Counter


WHITESPACE = '▁'
PAD = '<PAD>'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'


class MCounter(Counter):
    """This is a slight extention of the ``Collections.Counter`` class
    to also allow multiplication with integers.
    https://stackoverflow.com/a/74830621"""

    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError("Non-int factor")
        return MCounter({k: other * v for k, v in self.items()})

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return MCounter(super().__add__(other))
