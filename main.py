from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
# Make sure 'retriever' is defined in 'vector.py' as a function or variable.
# If 'retriever' is a function or class, import it accordingly.
# For example, if 'vector.py' contains 'def retriever(...):', this import is correct.
# If not, update 'vector.py' to define 'retriever', or fix the import as needed.
from vector import retriever  # Ensure 'retriever' exists in 'vector.py'


model = OllamaLLM(model="llama3.2")
template ="""
You are a helpful assistant. Answer the question based on the input provided.
Here are some reviews: {reviews}
Here is the question: {question}"""

prompt = ChatPromptTemplate.from_template(template)
chain= prompt | model
while True:
    print("\n\n------------------------------------------------------")
    question = input("Enter your question (or type 'exit' to quit): ")
    print("\n\n")
    if question.lower() == 'exit':
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
