from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate

class Llama:
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        """Loads the Llama model."""
        model = Ollama(model=self.model_name)
        return model

    def chat(self, question, template=None):
        if template == None:
            template = "{question}"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model
        reply = chain.invoke({"question": question})
        return reply
    
# Usage
if __name__ == "__main__":
    llama = Llama()
    question = "who are you?"
    template = """
        You are tasked with understanding and answering questions from diverse perspectives. Consider the original question carefully and generate multiple related queries that approach the topic from different angles. Aim to cover various interpretations and ensure broad retrieval coverage.

        Original question: {question}
        Related queries:
        1. 
        2. 
        3. 
    """
    reply = llama.chat(question, template)
    print(reply)
