"""
Original author(s):  Linus Åberg  <guslinuab@student.gu.se>

File purpose: Contains method to connect to database
"""

import sqlite3

def DB_Connection(db_name):
    ''' Returns a Database connection'''
    conn = sqlite3.connect(db_name)
    print("Connection to DB successfully")
    return conn