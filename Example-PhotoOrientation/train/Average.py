from functools import reduce

class Average:
    def __init__(self):
        self.items = list()

    def add(self, item):
        self.items.append(item)

    def calculate(self):
        return reduce(lambda x, y: x + y, self.items) / len(self.items)
