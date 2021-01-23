import os
from flask import render_template, request, json, Response
from flask import Flask, render_template,request,jsonify,redirect,url_for
from flask import redirect, url_for, request, render_template, send_file
from flask_wtf.file import FileField
from wtforms import SubmitField

class Config(object):
    SECRET_KY = os.environ.get('SECRET_KEY') or "secret_string"
    
class UploadForm(Form):
    file = FileField()
    submit = SubmitField("Submit")
    download = SubmitField("Download")
    introduction = SubmitField("here.")
    anotherpage = SubmitField("Another Page")
    samepage = SubmitField("Same Page")
    codepage = SubmitField("Page for Code.")
