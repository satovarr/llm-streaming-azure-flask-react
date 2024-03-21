from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from collections import deque
from dotenv import dotenv_values

config = dotenv_values()

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
        response = self.create().invoke(messages)
        return response.content


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

