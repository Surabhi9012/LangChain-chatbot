from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain.document_loaders import UnstructuredURLLoader, URLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import logging
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotManager:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.chain = None
        self.initialize_components()

    def load_documents(self):
        try:
            urls = ["https://brainlox.com/courses/category/technical"]
            loader = UnstructuredURLLoader(urls=urls)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            return text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def initialize_components(self):
        try:
            self.embeddings = OpenAIEmbeddings()
            
            documents = self.load_documents()
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./data"
            )
            
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=5,
                return_messages=True
            )
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo"),
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=memory,
                return_source_documents=True,
                verbose=True
            )
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise

    def get_response(self, question):
        try:
            response = self.chain({"question": question})
            return {
                "answer": response["answer"],
                "source_documents": [doc.page_content for doc in response["source_documents"]]
            }
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise

class ChatAPI(Resource):
    def __init__(self):
        self.chatbot = ChatbotManager()

    def post(self):
        try:
            data = request.get_json()
            
            if not data or 'question' not in data:
                return {"error": "Missing question in request"}, 400
                
            question = data['question']
            response = self.chatbot.get_response(question)
            
            return {
                "status": "success",
                "data": response
            }, 200
            
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return {"error": str(e)}, 500

app = Flask(__name__)
api = Api(app)

api.add_resource(ChatAPI, '/api/chat')

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)