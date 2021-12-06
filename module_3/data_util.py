import os
import glob


class DataFileGenerator(object):
    """generates all data paths and corresponding validation, test paths"""
    def __init__(self, path):
        self.path = path
        self.current = 0
        self.file = glob.glob(os.path.join(path, "train_c*_d*.csv"))
        self.file = sorted(self.file)
        print("total dataset.. ", len(self.file))
        self.high = len(self.file)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current == self.high:
            raise StopIteration
        self.current += 1
        return self.file[self.current - 1]

