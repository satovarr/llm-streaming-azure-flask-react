from flask import Flask, Response
from flask_cors import CORS, cross_origin
from app.llm import answer_general_question, get_mocked_context
import time

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



def store_message(message: str):
    print(f"\nmessage: {message} \nStored")


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
        for message in completion:
            yield f"data:{message}\n\n"
            print(f"message: {message}. compleion length: {len(completion)}")
        # store_message(completion)
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

