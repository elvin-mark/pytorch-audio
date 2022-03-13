from email.policy import default
from flask import Flask, render_template, request
from flask_socketio import SocketIO, send, emit
from argparse import ArgumentParser

app = Flask(__name__)
app.config["SECTRET_KEY"] = "dashboard_secret"
socketio = SocketIO(app)

sid = None


@app.route("/")
def home():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    global sid
    print("user connected")
    sid = request.sid


@app.route("/send_data", methods=["POST"])
def handle_data():
    if request.method == "POST":
        data = request.get_json()
        with app.test_request_context("/"):
            emit("data", data, namespace="/", room=sid)
    return "received!"


@app.route("/send_samples", methods=["POST"])
def handle_samples():
    if request.method == "POST":
        data = request.get_json()
        with app.test_request_context("/"):
            emit("samples", data, namespace="/", room=sid)
    return "received!"


@app.route("/send_model", methods=["POST"])
def handle_model():
    if request.method == "POST":
        data = request.get_json()
        with app.test_request_context("/"):
            emit("model", data, namespace="/", room=sid)
    return "received!"


if __name__ == "__main__":
    parser = ArgumentParser(
        description="cli for initialize the server for the dashboard")
    parser.add_argument("--port", type=int, default=12345,
                        help="port for the server")
    args = parser.parse_args()
    socketio.run(app, port=args.port)
