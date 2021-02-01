import re


class Article:
    sentence_separator = r"(?<!\w[\.]\w[\.])(?<![A-Z][a-z][\.])(?<=[\.]|\?|\!)\s"

    def __init__(self, langid: str):
        self.langid = langid
        self.text = None
        self.title = None
        self.contents = None
        self.sentences = []

    def from_text(self, text: str):
        if text is None:
            raise TypeError
        elif text == "":
            raise ValueError('The input text of the article must not be empty!')
        self.text = text
        self.__detect_title()
        self.__split()

    def from_sentences(self, sentences: list):
        if all(isinstance(sentence, str) for sentence in sentences):
            self.title = sentences[0]
            self.contents = ' '.join(sentences[1:])
            self.text = self.title + '\n' + self.contents
            self.sentences = sentences
        else:
            raise TypeError('The type of a sentence in the sentence list is not string')

    def __detect_title(self):
        parts = self.text.split("\n", 1)
        if len(parts) == 1:
            # Title does not exist
            self.contents = parts[0].strip()
        elif len(parts) == 2:
            # Title exists
            self.title = parts[0].strip()
            self.contents = parts[1].strip()
            self.sentences.append(self.title)
        else:
            # Empty article
            raise ValueError
        self.contents = self.contents.replace('\n', ' ')

    def __split(self):
        self.sentences += re.split(self.sentence_separator, self.contents)
        if '' in self.sentences:
            self.sentences.remove('')

    def __str__(self):
        return self.text
