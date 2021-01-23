from flask import Flask

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"   ##we use this whenever we use wt forms


from application import routes



