from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


CHROMA_PATH="chroma"
embedder=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

PROMPT_TEMPLATE =
"""
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

if __name__=="__main__":
    query_text = "where is alice"

    db=Chroma(persist_directory=CHROMA_PATH, embedding_function=embedder)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    model = ChatOpenAI()
    response_text = model.predict(prompt)
    print(f"---------------------final response------------------")
    print(response_text)

