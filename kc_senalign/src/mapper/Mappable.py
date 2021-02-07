from abc import abstractmethod


class Mappable:

    @abstractmethod
    def map_sentences(self) -> list:
        pass
