from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

#DATA_PATH="/input/ragdata"
DATA_PATH="."
CHROMA_PATH="chroma"


def load_docs() -> list:
    """
    Loads the documetns from the directory specified  using DirectoryLoader

    Returns
        list: a list of the documentss loaded in the directory , this example i am working with has one single txt file so one doc
    """
    loader=DirectoryLoader(DATA_PATH,glob="*.txt")
    documents=loader.load()
    return documents


def split_text(doc:list) -> list:
    """
    Takes the documetns and splits it to chunks accordin to the static parameters specifed 
    in RecursiveCharacterTextSplitter then return the chunks
     
    Returns 
       list: a list of teh cotent of the doc split to chunks
    
    """
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,)
    chunks=text_splitter.split_documents(doc)
    return chunks


def create_db(chunks,embedding_model):
    """
    Creates teh db and saves the embedded chunks to it , for embedding model we pass it to have more flxiiblt if wanna use it later

    """
    db=Chroma.from_documents(
        chunks,embedding_model,persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} to the db {CHROMA_PATH}")


if __name__=="__main__":
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    doc=load_docs()
    chunks=split_text(doc)
    create_db(chunks,embedding_model)



