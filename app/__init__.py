from flask import Flask, Response
from flask_cors import CORS, cross_origin
from app.llm import answer_general_question, get_mocked_context, answer_dqa_question
import time

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



def store_message(message: str):
    print(f"\nmessage: {message[:15]} \nStored")


@app.route("/stream-chat")
@cross_origin()
def stream_chat():
    start_time = time.time()
    def generate():
        completion = answer_general_question(
            question="Can you please give me a brief history of the United States?",
            context=get_mocked_context(),
            streaming=True
        )
        message = ""
        for chunk in completion:
            message += chunk.content
            yield f"data:{chunk.content}\n\n"
        store_message(message)
        print(f"Time taken: {time.time() - start_time}")
        yield "data: CLOSE\n\n"

    return Response(generate(), content_type="text/event-stream")


@app.route('/stream-dqa')
def stream_dqa():
    start_time = time.time()
    def generate():
        completion, retriever = answer_dqa_question(
            question="What can I ask about?",
            context=get_mocked_context(),
            streaming=True
        )
        message = ""
        for chunk in completion:
            message += chunk
            yield f"data:{chunk}\n\n"
        yield f"data: {retriever.metadata_storage}\n\n"
        store_message(message)
        print(f"Time taken: {time.time() - start_time}")
        yield "data: CLOSE\n\n"

    return Response(generate(), content_type="text/event-stream")

def generate_mock():
    tokens = "Hello, this is a mocked chatbot. How can I help you?".split(" ")
    for token in tokens:
        time.sleep(0.5)
        yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"

@app.route('/stream-example')
def stream():
    return Response(generate_mock(), mimetype='text/event-stream')


@app.route("/chat")
@cross_origin()
def chat():
    start_time = time.time()
    completion = answer_general_question(
        question="What is the capital of France?",
        context=get_mocked_context(),
        streaming=False
    )
    store_message(completion)
    print(f"Time taken: {time.time() - start_time}")
    return completion

