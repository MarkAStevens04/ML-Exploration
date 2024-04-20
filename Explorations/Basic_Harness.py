from Explorations.Abstract_Harness import AbstractHarness
from Data.Generation.Abstract_DataGen import AbstractDataGen
from Data.Generation.Basic_DataGen import BasicGen


class BasicHarness(AbstractHarness):
    def __init__(self):
        pass









if __name__ == '__main__':
    dg = BasicGen()
    df = BasicGen.generate()


    bh = BasicHarness

