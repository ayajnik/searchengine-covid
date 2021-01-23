import sqlite3
import sqlalchemy
import pandas as pd
#import pyodbc


def sql_execute(command):

    a = cur.execute(command)
    return a

def connect_to_sql():
 
    global con
    global cur
    
    con = sqlite3.connect('walletexplorer.db')
    cur = con.cursor()

df = pd.read_csv("D:\\searchengine-covid\\article_info.csv")
df.to_sql('Article_info_new', con = con)








