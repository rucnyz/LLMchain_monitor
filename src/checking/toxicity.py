from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


class ToxicityChecker:
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.pipeline = TextClassificationPipeline(
            model = self.model, tokenizer = self.tokenizer, device = self.device
        )

    def check(self, text: str) -> float:
        result = self.pipeline(text, truncation = True, max_length = self.tokenizer.model_max_length)
        return result[0]["score"] if result[0]["label"] == "toxic" else 1 - result[0]["score"]
