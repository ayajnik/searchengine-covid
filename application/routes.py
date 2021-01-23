from application import app
from flask import render_template, request, json, Response
from flask import Flask, render_template,request,jsonify,redirect,url_for
from flask import redirect, url_for, request, render_template, send_file
from flask_wtf.file import FileField
from wtforms import SubmitField
    #import flask_api
from flask_wtf import Form
import bm25
from bm25 import print_answers

@app.route("/")
@app.route('/introduction', methods=['GET','POST'])
def introduction():

    form = UploadForm()
    if request.method == "POST":
        if form.validate_on_submit():
            file_name_1 = form.file.data

            return redirect(url_for("index.html", form=form))

    return render_template("intro_1.html", form=form)

@app.route('/samepage', methods=['GET','POST'])
def samepage():

    form = UploadForm()
    if request.method == "POST":
        if form.validate_on_submit():
            file_name_2 = form.file.data

            return redirect(url_for("index.html", form=form))

    return render_template("prac_1.html", form=form)

@app.route('/anotherpage', methods=['GET','POST'])
def anotherpage():

    form = UploadForm()
    if request.method == "POST":
        if form.validate_on_submit():
            file_name_3 = form.file.data

            return redirect(url_for("index.html", form=form))

    return render_template("prac_1.html", form=form)


#
@app.route("/index")
def index():
    form = UploadForm()
    ##when the button is pressed, it will give us a POST method
    if request.method == "POST":
        if form.validate_on_submit():
            file_name = form.file.data
            database(name=file_name.filename, data=file_name.read())
            ##print("FILE : ".format(file_name.filename))
            '''
            For rendering results on HTML GUI
            '''
            int_features = [str(x) for x in request.form.values()]
            words = print_answers(int_features)  # checking the vocabulary

            print('\n')
            xy = render_template('index.html', prediction_text='The most similar words are : {}'.format(words))
            print('\n')
            print('The total number of similar words, ', len(words))
            print('\n')
            return xy

            
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = print_answers(data)

    output = prediction[0]
    return jsonify(output)



class UploadForm(Form):
    file = FileField()
    submit = SubmitField("Submit")
    download = SubmitField("Download")
    introduction = SubmitField("here.")
    anotherpage = SubmitField("Another Page")
    samepage = SubmitField("Same Page")
    codepage = SubmitField("Page for Code.")
