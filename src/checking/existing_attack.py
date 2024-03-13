import os.path
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

existing_embeddings_path = "../dataset/"
jailbreak_embeddings_path = os.path.join(existing_embeddings_path, "jailbreak_embeddings.npy")


class ExistingAttackChecker:
    def __init__(self, embeddings_path: str):
        # TODO need to add other options
        # TODO need to add which attack is considered
        self.model = SentenceTransformer(embeddings_path)
        if os.path.exists(jailbreak_embeddings_path):
            self.existing_embeddings = {"jailbreak": np.load(jailbreak_embeddings_path)}
        else:
            self.existing_embeddings = self.process_harm_input()

    def check(self, text: str) -> bool:
        target_embeddings = self.model.encode([text])
        # TODO here we only consider one sample a time
        for attack_type, attack_embeddings in self.existing_embeddings.items():
            score = torch.max(util.cos_sim(target_embeddings[:], attack_embeddings[:]), dim = 1)[0].item()
            if score > 0.80:
                return True
        return False

    def process_harm_input(self):
        # jailbreaking
        jailbreak_sample_path = "../dataset/jailbreak/jailbreak_question.csv"
        jailbreak_samples = pd.read_csv(jailbreak_sample_path, header = None)[0].to_list()
        jailbreak_embeddings = self.model.encode(jailbreak_samples)
        np.save(jailbreak_embeddings_path, jailbreak_embeddings)
        # TODO need to add other attacks
        return {"jailbreak": jailbreak_embeddings}
