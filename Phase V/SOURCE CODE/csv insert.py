import sqlite3
import csv

# Define the path to your CSV file
csv_file_path = 'test.csv'

# Establish a connection to the SQLite database
conn = sqlite3.connect('dataset.db')
cursor = conn.cursor()

# Open the CSV file and read its contents
with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)

    # Get the column names from the first row (header)
    headers = next(reader)

    # Create the table based on the CSV columns
    columns = ', '.join([f'{header} TEXT' for header in headers])
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS Movies ({columns})
    ''')

    # Insert the rows from the CSV file into the table
    placeholders = ', '.join(['?'] * len(headers))  # Create placeholders for the values
    cursor.executemany(f'''
        INSERT INTO Movies ({', '.join(headers)})
        VALUES ({placeholders})
    ''', reader)

# Commit the changes and close the connection
conn.commit()
conn.close()
