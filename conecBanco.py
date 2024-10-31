import mysql.connector


# abrindo coneccao banco mysql

conn = mysql.connector.connect(
    host ='localhost' ,
    user = 'root',
    password ='MySon2021#' ,
    database = 'water_monitoring'
)

cursor = conn.cursor()
# exemplo de consulta

cursor.execute("SELECT * FROM images")
rows = cursor.fetchall()

for row in rows:
   print(row)

cursor.close()
conn.close()
