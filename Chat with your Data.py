# Import necessary libraries
import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# Load PDF document using PyPDFLoader
loader = PyPDFLoader("path to your document")
pages = loader.load()


# Split the loaded document into smaller chunks for processing
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=900,
    chunk_overlap=150,
    length_function=len,
)

splits = text_splitter.split_documents(pages)


# Set up OpenAI embeddings and create a vector database
os.environ["OPENAI_API_KEY"] = "Enter your API Key"

embeddings = OpenAIEmbeddings(disallowed_special=())

vectorDb = Chroma.from_documents(
    embedding=embeddings,
    documents=splits,
    persist_directory="/docs/chroma"
)
vectorDb.persist()


# Read the prompt template from a file
prompt_file_path = "path to your template file"
with open(prompt_file_path, "r") as file:
    prompt_template = file.read()

# Define Prompt Template
PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question", "context"]
    )


# Initialize Chat OpenAI model
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
global qa


# Initialize RetrievalQA object for question answering
qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vectorDb.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
    )


# Define chat function for handling user queries
def chat(chat_history, query):
    res = qa.run(query)
    progressive_response = ""

    for ele in "".join(res):
        progressive_response += ele + ""
        yield chat_history + [(query, progressive_response)]


# Gradio Interface Setup
with gr.Blocks() as demo:
    gr.HTML(
        """<h1>Welcome to AI PDF Assistant</h1>"""
    )
    gr.Markdown(
        "AI Assistant for Physics<br>"
        "Type your query, and  hit enter. Feel free to explore and ask questions! <br> "
        "Click on 'Clear Chat History' to delete all previous conversations.<br>"
        "Happy exploring!"
    )
    with gr.Tab("AI Assistant"):

         # Define chatbot interface elements
        chatbot = gr.Chatbot()
        query = gr.Textbox(
            label="Type your query here, then press 'enter' and scroll up for response"
        )
        clear = gr.Button("Clear Chat History!")

        # Define callback for query submission
        query.submit(chat, [chatbot, query], chatbot)

        # Define callback for clearing chat history
        clear.click(lambda: None, None, chatbot, queue=False)


# Launch Gradio interface
demo.queue().launch()


