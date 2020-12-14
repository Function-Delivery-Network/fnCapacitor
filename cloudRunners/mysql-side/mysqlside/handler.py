import mysql.connector
import os, requests, json

def handle(req):

    host = os.getenv("mysql_host", "127.0.0.1")
    port = os.getenv("mysql_port", "3306")
    user = os.getenv("mysql_user", "root")
    database = os.getenv("mysql_database", "demo")


    with open("/var/openfaas/secrets/secret-mysql-key") as f:
        password = f.read().strip()

    cnx = mysql.connector.connect(host=host, user=user, password=password, port=port)

    cursor = cnx.cursor()

    cnx.database = database

    cursor.execute("SELECT * FROM minio")

    res = cursor.fetchall()

    rowHeaders = [x[0] for x in cursor.description]

    for row in res:
        jsonRes = dict(zip(rowHeaders,row))

    cursor.close()

    return json.dumps(jsonRes)