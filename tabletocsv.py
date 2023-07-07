import pyodbc
import csv

# Connection details
server = 'VN-PF3DT8K5'
database = 'DW_STOCK'
username = 'int-dba'
password = 'vm1dta12#$'
table = 'ACB_Trade'

# Establishing a connection to the SQL Server database
conn = pyodbc.connect(
    f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
)

# Creating a cursor to execute SQL queries
cursor = conn.cursor()

# Fetching data from the specified table
cursor.execute(f'SELECT * FROM {table}')
rows = cursor.fetchall()

# Getting the column names
column_names = [column[0] for column in cursor.description]

# Defining the output CSV file path
output_file = 'ACB_Trade.csv'

# Writing data to the CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Writing the column names as the header
    writer.writerow(column_names)
    
    # Writing the data rows
    writer.writerows(rows)

print(f'Table {table} successfully exported to {output_file}.')
