
class SolarImage:
    """
    Representation of a solar image.
    """
    def __init__(self, data, header=None):
        self.original = data.copy()
        self.data = data.copy()
        self.header = header
        self.history = []

    def reset(self):
        self.data = self.original.copy()
        self.history.append("reset")

    def copy(self):
        return SolarImage(self.data.copy(), self.header)
