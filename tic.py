import time

class Tic():
    ttime = 0

    @classmethod
    def tic(cls):
        cls.ttime = time.time()

    @classmethod
    def toc(cls):
        print(time.time() - cls.ttime)