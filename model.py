from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

class PromptProvider:
    def __init__(self, template=None):
        self.template = template or (
            '''Use the following pieces of information to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext: {context}\nQuestion: {question}\n\nOnly return the helpful answer below and nothing else.\nHelpful answer:\n'''
        )
    def get_prompt(self):
        return PromptTemplate(template=self.template, input_variables=['context', 'question'])

class LLMProvider:
    def __init__(self, model_name="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama", max_new_tokens=512, temperature=0.5):
        self.model_name = model_name
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
    def load(self):
        return CTransformers(
            model=self.model_name,
            model_type=self.model_type,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature
        )

class EmbeddingsProvider:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        self.model_name = model_name
        self.device = device
    def get(self):
        return HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={'device': self.device})

class VectorDBProvider:
    def __init__(self, db_path=DB_FAISS_PATH, embeddings_provider=None):
        self.db_path = db_path
        self.embeddings_provider = embeddings_provider or EmbeddingsProvider()
    def load(self):
        embeddings = self.embeddings_provider.get()
        return FAISS.load_local(self.db_path, embeddings)

class QABot:
    def __init__(self, llm_provider=None, prompt_provider=None, db_provider=None):
        self.llm_provider = llm_provider or LLMProvider()
        self.prompt_provider = prompt_provider or PromptProvider()
        self.db_provider = db_provider or VectorDBProvider()
        self._qa_chain = None
    def build_chain(self):
        llm = self.llm_provider.load()
        prompt = self.prompt_provider.get_prompt()
        db = self.db_provider.load()
        self._qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
    def answer(self, query):
        if self._qa_chain is None:
            self.build_chain()
        return self._qa_chain({'query': query})

# Chainlit integration decoupled from core logic
class ChainlitHandler:
    def __init__(self, bot=None):
        self.bot = bot or QABot()
    async def on_chat_start(self):
        self.bot.build_chain()
        msg = cl.Message(content="Starting the bot...")
        await msg.send()
        msg.content = "Hi, Welcome to Medical Bot. What is your query?"
        await msg.update()
        cl.user_session.set("chain", self.bot)
    async def on_message(self, message: cl.Message):
        bot = cl.user_session.get("chain")
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True
        res = await bot._qa_chain.acall(message.content, callbacks=[cb])
        answer = res["result"]
        sources = res["source_documents"]
        if sources:
            answer += f"\nSources:" + str(sources)
        else:
            answer += "\nNo sources found"
        await cl.Message(content=answer).send()

# Chainlit event registration
handler = ChainlitHandler()

@cl.on_chat_start
async def start():
    await handler.on_chat_start()

@cl.on_message
async def main(message: cl.Message):
    await handler.on_message(message) 