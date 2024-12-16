from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class QuestionDetector:
    def __init__(self, model_name="mrsinghania/asr-question-detection"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Function to predict if a sentence is a question
    def is_question(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class == 1

if __name__ == '__main__':
    qd = QuestionDetector()
    # Test the function
    sentence = "is this a dog"
    print(f"'{sentence}' is a question: {qd.is_question(sentence)}")

    sentence = "dog"
    print(f"'{sentence}' is a question: {qd.is_question(sentence)}")
