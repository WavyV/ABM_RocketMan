import json
import os
import threading

from flask import Flask, request
from termcolor import colored

app = Flask(__name__, static_folder="dist")

dir_path = os.path.dirname(os.path.realpath(__file__))
form_path = os.path.join(dir_path, "form.json")

# Create data as arrays if not exists.
if not os.path.isfile(form_path):
    with open(form_path, "w") as f:
        json.dump(dict(), f)

# Load data into memory.
with open(form_path) as f:
    FORM_DATA = json.load(f)


def save_data():
    """Save data every 5 seconds."""
    threading.Timer(5, save_data).start()
    with open(form_path, "w") as f:
        json.dump(FORM_DATA, f, indent=4, sort_keys=True)
    print(colored("Saved data", "green"))


@app.route("/")
def root():
    return app.send_static_file("index.html")


@app.route("/bundle.js")
def bundle():
    return app.send_static_file("bundle.js")


@app.route("/done", methods=["POST"])
def done():
    userdata = request.get_json()
    userid = userdata.get('userid', False)
    print(userdata)
    if userid:
        FORM_DATA[userid] = userdata
        print("Form for user {0}".format(userid))
    print(colored("Form without ID", "red"))
    return ""


if __name__ == "__main__":
    save_data()
    app.run(threaded=True)
