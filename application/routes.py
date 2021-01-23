from application import app
from flask import render_template, request, json, Response
from flask import Flask, render_template,request,jsonify,redirect,url_for
from flask import redirect, url_for, request, render_template, send_file
from flask_wtf.file import FileField
from wtforms import SubmitField
    #import flask_api
from flask_wtf import Form


@app.route('/introduction', methods=['GET','POST'])
def introduction():

    form = UploadForm()
    if request.method == "POST":
        if form.validate_on_submit():
            file_name_1 = form.file.data

            return redirect(url_for("complete.html", form=form))

    return render_template("intro_1.html", form=form)

@app.route('/samepage', methods=['GET','POST'])
def samepage():

    form = UploadForm()
    if request.method == "POST":
        if form.validate_on_submit():
            file_name_2 = form.file.data

            return redirect(url_for("complete.html", form=form))

    return render_template("prac_1.html", form=form)

@app.route('/anotherpage', methods=['GET','POST'])
def anotherpage():

    form = UploadForm()
    if request.method == "POST":
        if form.validate_on_submit():
            file_name_3 = form.file.data

            return redirect(url_for("complete.html", form=form))

    return render_template("prac_1.html", form=form)


#@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", index=True )


class UploadForm(Form):
    file = FileField()
    submit = SubmitField("Submit")
    download = SubmitField("Download")
    introduction = SubmitField("here.")
    anotherpage = SubmitField("Another Page")
    samepage = SubmitField("Same Page")
    codepage = SubmitField("Page for Code.")
