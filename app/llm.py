from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from collections import deque
from dotenv import dotenv_values
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from dotenv import dotenv_values
from typing import Iterator
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from langchain_core.messages import BaseMessageChunk
from azure.search.documents.indexes.models import (
    ComplexField,
    CorsOptions,
    SearchIndex,
    ScoringProfile,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
)


from dotenv import dotenv_values
import os
import uuid

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient


from app.custom_retriever import CustomRetriever


config = dotenv_values()

def processed_file(pdf_file_path, project_id="test"):
    file = pdf_file_path
    try:
        # Open the PDF file in binary mode
        with open(file, "rb") as pdf_file:
            # Read the file content
            pdf_content = pdf_file.read()

            # Pass the binary content to Azure's function
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-layout", pdf_content
            )
            result = poller.result()

        page_number = 1

        list_pages = []

        for page in result.pages:
            page_uuid = str(uuid.uuid4())
            page_text = ""

            for line in page.lines:
                page_text += line.content
                page_text += "\n"

            name_combined_txt = (
                f"{os.path.splitext(file)[0]}_page_{str(page_number)}.txt"
            )
            metadata = {
                "page_uuid": page_uuid,
                "page_number": page_number,
                "document_name": file,
                "project_id": project_id,
            }
            list_pages.append({"page_text": page_text, "metadata": metadata})

            page_number += 1
        return list_pages

   
    except Exception as e:
        print(f"Error: {e}")

def check_and_create_index(index_name):
    os.environ["AZURE_COGNITIVE_SEARCH_INDEX_NAME"] = index_name
    service_name = "as-eus2-dev-ia"
    admin_key = config["ADMIN_KEY"]
    os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"] = service_name
    os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"] = admin_key
    # Create an SDK client

    endpoint = "https://{}.search.windows.net/".format(service_name)
    admin_client = SearchIndexClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(admin_key),
    )

    search_client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(admin_key),
    )

    # Delete the index if it exists
    try:
        result = admin_client.delete_index(index_name)
        # logger.debug('Index', index_name, 'Deleted')
    except Exception as ex:
        return False

    fields = [
        SimpleField(name="project_id", type=SearchFieldDataType.String),
        SimpleField(name="page_uuid", type=SearchFieldDataType.String, key=True),
        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="document_name",
            type=SearchFieldDataType.String,
            facetable=True,
            filterable=True,
            sortable=True,
        ),
        SearchableField(
            name="page_text", type=SearchFieldDataType.String, analyzer_name="en.lucene"
        ),
    ]

    cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
    scoring_profiles = []

    # Request schema creation
    index = SearchIndex(
        name=index_name,
        fields=fields,
        scoring_profiles=scoring_profiles,
        cors_options=cors_options,
    )

    try:
        result = admin_client.create_index(index)
    except Exception as ex:
        return False
    return search_client

def index_documents_azure_ai_search(search_client, list_doc_pages):
    list_documents = []
    for document in list_doc_pages:
        metadata = document.get("metadata", {})  # Extract metadata dictionary
        for key, value in metadata.items():
            document[key] = value
        del document["metadata"]
        document["@search.action"] = "upload"
        list_documents.append(document)

    try:
        result = search_client.upload_documents(documents=list_documents)
    except Exception as ex:
        return False
    return True



document_analysis_client = DocumentAnalysisClient(
        endpoint=config["ENDPOINT"], credential=AzureKeyCredential(config["API_KEY"])
    )

pdf_file_path = "/Users/santiagotovar/code/llm-streaming-azure-flask-react/app/data/processed.pdf"
project_id = "stream-chat-example-dqa"
list_doc_pages = processed_file(pdf_file_path, project_id)

# File indexation
search_client = check_and_create_index(index_name=project_id)

index_documents_azure_ai_search(search_client, list_doc_pages)



class LLMModel:
    def __init__(
        self,
        open_api_version,
        open_api_key,
        azure_endpoint,
        deployment_name,
        model,
        temperature,
        streaming: bool,
    ):
        self.open_api_version = open_api_version
        self.open_api_key = open_api_key
        self.azure_endpoint = azure_endpoint
        self.deployment_name = deployment_name
        self.model = model
        self.temperature = temperature
        self.streaming = streaming

    def create(self):
        llm_instance = AzureChatOpenAI(
            openai_api_version=self.open_api_version,
            openai_api_key=self.open_api_key,
            azure_endpoint=self.azure_endpoint,
            deployment_name=self.deployment_name,
            model=self.model,
            temperature=self.temperature,
            streaming=True,
        )
        return llm_instance

    def call_llm(self, messages: list):
        response = self.create().stream(messages)
        return response


class GeneralChatConversation(LLMModel):
    def __init__(
        self,
        open_api_version,
        open_api_key,
        azure_endpoint,
        deployment_name,
        model,
        temperature,
        question: str,
        context: deque,
        streaming: bool,
    ):
        super().__init__(
            open_api_version,
            open_api_key,
            azure_endpoint,
            deployment_name,
            model,
            temperature,
            streaming,
        )
        self.question = question
        self.context = context

    @property
    def general_chat_content(self):
        _content = """You are a helpful, respectful and honest assistant.
        Always answer as helpfully as possible, while being safe.
        Your answers should not include any harmful, unethical, racist,
        sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent,
        explain why instead of answering something not correct.
        If you don't know the answer to a question, please don't
        share false information."""
        return _content

    def generate_chat(self):
        context_messages = list(self.context)
        messages = [SystemMessage(content=self.general_chat_content)]
        for item in context_messages:
            messages.append(HumanMessage(content=item[0]))
            messages.append(AIMessage(content=item[1]))
        messages.append(HumanMessage(content=self.question))
        results_content = self.call_llm(messages)
        self.context.append((self.question, results_content))
        return results_content

class DQAConversation(LLMModel):

    def __init__(
        self,
        open_api_version,
        open_api_key,
        azure_endpoint,
        deployment_name,
        model,
        temperature,
        question: str,
        context: deque,
        streaming: bool,
    ):
        super().__init__(
            open_api_version,
            open_api_key,
            azure_endpoint,
            deployment_name,
            model,
            temperature,
            streaming,
        )
        self.question = question
        self.context = context
        self.metadata_storage = []
        self.retriever = CustomRetriever(
        content_key="page_text", top_k=5
    )
        self.retriever.metadata_storage = self.metadata_storage

    def get_langchain_pipeline(self):
        """
        Constructs and returns a LangChain pipeline integrating a language model and a retriever.

        This function sets up a LangChain pipeline designed for question answering. It combines a given
        language model and a retriever to process queries. The pipeline uses a predefined template to format
        responses, ensuring that each answer is based solely on the context provided by the retriever and
        includes a list of document names used for answering.

        Parameters:
        model (AzureChatOpenAI): The language model used for generating answers.
        retriever (Chroma): The retriever used for fetching relevant context based on the input question.

        Returns:
        A LangChain pipeline configured for processing and answering questions with context and document references.

        Example:
        langchain_pipeline = get_langchain_pipeline(model, retriever)
        answer = langchain_pipeline({"question": "What is AI?"})
        """
        template = """You are a helpful assistant designed to output a string. Answer the question based only on the following context:
        {context}

        Question: {question}
        Provide an answer in string format
        """

        prompt = ChatPromptTemplate.from_template(
            template, response_format={"type": "string"}
        )
        string_parser = StrOutputParser()

        chain = (
            {
                "context": itemgetter("question") | self.retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | self.create()
            | string_parser
        )
        return chain

    def call_llm(self, messages: list) -> tuple[Iterator[BaseMessageChunk], list]:
        response = self.get_langchain_pipeline.stream(messages)
        return response, self.metadata_storage



def get_mocked_context():
    return deque([("Hello", "Hi there!"), ("How are you?", "I'm doing great!")])

def answer_general_question(question, context, streaming: bool):
    llm_instance = GeneralChatConversation(
        open_api_version="2023-07-01-preview",
        open_api_key=config["OPENAI_API_KEY"],
        azure_endpoint=config["AZURE_ENDPOINT"],
        deployment_name=config["CONFIG_DEPLOYMENT_NAME"],
        model="gpt-35-turbo",
        temperature=0,
        question=question,
        context=context,
        streaming=streaming,
    )
    answer = llm_instance.generate_chat()
    return answer

def answer_dqa_question(question, context, streaming: bool):
    llm_instance = DQAConversation(
        open_api_version="2023-07-01-preview",
        open_api_key=config["OPENAI_API_KEY"],
        azure_endpoint=config["AZURE_ENDPOINT"],
        deployment_name=config["CONFIG_DEPLOYMENT_NAME"],
        model="gpt-35-turbo",
        temperature=0,
        question=question,
        context=context,
        streaming=streaming,
    )
    answer, metadata_storage = llm_instance.call_llm(context)
    return answer, metadata_storage