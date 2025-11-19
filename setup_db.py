import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect('barber_appointments.db')
cursor = conn.cursor()

cursor.execute('''
    DROP TABLE IF EXISTS appointments
''')

cursor.execute('''
    CREATE TABLE appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name TEXT NOT NULL,
        phone_number TEXT,
        appointment_date TEXT NOT NULL,
        appointment_time TEXT NOT NULL,
        service_type TEXT NOT NULL,
        barber_name TEXT NOT NULL,
        duration_minutes INTEGER,
        price REAL,
        status TEXT DEFAULT 'confirmed',
        notes TEXT
    )
''')

base_date = datetime.now()
appointments = [
    ("John Smith", "555-0101", (base_date + timedelta(days=1)).strftime("%Y-%m-%d"), "10:00 AM", "Premium Haircut", "Marco", 30, 35, "confirmed", "Regular customer"),
    ("Sarah Johnson", "555-0102", (base_date + timedelta(days=1)).strftime("%Y-%m-%d"), "11:00 AM", "Hair Styling", "Alex", 45, 40, "confirmed", None),
    ("Mike Davis", "555-0103", (base_date + timedelta(days=2)).strftime("%Y-%m-%d"), "2:00 PM", "Beard Trim & Shape", "Tommy", 20, 25, "confirmed", "First time"),
    ("Emma Wilson", "555-0104", (base_date + timedelta(days=3)).strftime("%Y-%m-%d"), "9:00 AM", "Hot Towel Shave", "Marco", 30, 50, "confirmed", None),
    ("James Brown", "555-0105", (base_date + timedelta(days=4)).strftime("%Y-%m-%d"), "3:30 PM", "Fade Service", "Alex", 25, 30, "pending", "Awaiting confirmation"),
]

cursor.executemany('''
    INSERT INTO appointments 
    (customer_name, phone_number, appointment_date, appointment_time, service_type, barber_name, duration_minutes, price, status, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', appointments)

conn.commit()
conn.close()

print("Database created with sample appointments.")
