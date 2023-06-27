import os
from flask import Flask, jsonify, request
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
# from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI, ChatAnthropic
import faiss
from agent import AutoGPT

app = Flask(__name__)

@app.route('/research', methods=['POST'])
def do_research():
    keyword = request.json.get('keyword', '')

    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty

    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    agent = AutoGPT.from_llm_and_tools(
    ai_name="AutoResearch",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    # llm=ChatAnthropic(temperature=0),
    memory=vectorstore.as_retriever(),
    )
    # Set verbose to be true
    agent.chain.verbose = True

    result = agent.run([f"write a witty, humorous but concise report about {keyword}", f"save the report in the `report` directory"], limit=4)

    return jsonify({'status':'success', 'result': result})

@app.route('/reports', methods=['GET'])
def list_reports():
    reports = os.listdir('report')  # replace 'report' with the path to your report directory
    return jsonify({'status': 'success', 'result': reports})

@app.route('/reports/<report_name>', methods=['GET'])
def read_report(report_name):
    report_path = os.path.join('report', report_name)  # replace 'report' with the path to your report directory
    if os.path.exists(report_path):
        with open(report_path, 'r') as file:
            content = file.read()
        return jsonify({'status':'success', 'result': content})
    else:
        return jsonify({'status':'failed', 'error': 'File not found'}), 404

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
