from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route("/login")
@cross_origin(supports_credentials=True)
def login():
  return jsonify({'success': 'ok'})
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
            question="What is the capital of France?",
            context=get_mocked_context(),
            streaming=True
        )
        for message in completion:
            yield f"data:{message}\n\n"
        store_message(completion)
        print(f"Time taken: {time.time() - start_time}")
        yield "data: CLOSE\n\n"

    return Response(generate(), content_type="text/event-stream")


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

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8000, debug=True)