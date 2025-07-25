from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

class PDFLoader:
    def __init__(self, data_path, glob_pattern='*.pdf'):
        self.data_path = data_path
        self.glob_pattern = glob_pattern
    def load(self):
        loader = DirectoryLoader(self.data_path, glob=self.glob_pattern, loader_cls=PyPDFLoader)
        return loader.load()

class TextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def split(self, documents):
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return splitter.split_documents(documents)

class EmbeddingsProvider:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
        self.model_name = model_name
        self.device = device
    def get(self):
        return HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={'device': self.device})

class VectorDBBuilder:
    def __init__(self, db_path, embeddings_provider=None):
        self.db_path = db_path
        self.embeddings_provider = embeddings_provider or EmbeddingsProvider()
    def build_and_save(self, texts):
        embeddings = self.embeddings_provider.get()
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(self.db_path)

# Orchestrator for ingestion
class IngestPipeline:
    def __init__(self, loader=None, splitter=None, db_builder=None):
        self.loader = loader or PDFLoader(DATA_PATH)
        self.splitter = splitter or TextSplitter()
        self.db_builder = db_builder or VectorDBBuilder(DB_FAISS_PATH)
    def run(self):
        documents = self.loader.load()
        texts = self.splitter.split(documents)
        self.db_builder.build_and_save(texts)

if __name__ == "__main__":
    pipeline = IngestPipeline()
    pipeline.run() 