from flask import Flask

app = Flask('ping')

@app.route("/ping")
def ping_route():
    return ping()  # calls the function we defined

app.run(debug=True, host = '0.0.0.0')


def ping():
    retun 'PONG'