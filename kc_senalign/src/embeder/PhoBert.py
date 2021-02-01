import torch
from transformers import AutoTokenizer, AutoModel
from vncorenlp import VnCoreNLP
from src.embeder.EmbeddingModel import EmbeddingModel

# VnCoreNLP_JAR_ABSOLUTE_PATH = "/home/domh/PycharmProjects/KC-4.0/kc_senalign/lib/vncorenlp/VnCoreNLP-1.1.1.jar"
from src.sentence.Sentence import Sentence

VnCoreNLP_JAR_PATH = "lib/vncorenlp/VnCoreNLP-1.1.1.jar"


class PhoBert(EmbeddingModel):
    def __init__(self, device: torch.device):
        # print(os.getcwd())
        self.__device = device
        # print(device)
        self.__rdrsegmenter = VnCoreNLP(VnCoreNLP_JAR_PATH, annotators="wseg", max_heap_size='-Xmx500m')
        self.__tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.__model = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states=True).to(self.__device)

    def embed_sentence(self, sentence: Sentence) -> torch.Tensor:
        """
        Tokenize and embed a sentence with PhoBERT
        """
        try:
            line = self.__tokenize(sentence.text)
            # mapping words and their ids in vncorenlp dictionary
            input_ids = torch.tensor([self.__tokenizer.encode(line)])
            input_ids.to(self.__device)
            with torch.no_grad():
                features = self.__model(input_ids)
            # cleanup
            # del input_ids
            return self.__to_embedding(features)
        except:
            print("Maybe the input has more than 1 sentence. Please input ONE sentence only!")

    def __tokenize(self, text: str):
        """
        To perform word segmentation
        """
        segments = self.__rdrsegmenter.tokenize(text)
        return " ".join(segments[0])

    def __to_embedding(self, features):
        """
        Convert features to sentence embedding
        """
        hidden_states = features[2]
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        return torch.mean(
            torch.cat(
                tuple(last_four_layers),
                dim=-1
            ),
            dim=1
        ).squeeze()
