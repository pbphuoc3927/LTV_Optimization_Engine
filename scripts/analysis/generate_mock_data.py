import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

faker = Faker()

def generate_customers(n=50000):
    customers = []
    for i in range(n):
        customers.append({
            "customer_id": i+1,
            "name": faker.name(),
            "age": random.randint(18,70),
            "gender": random.choice(["M","F"]),
            "region": random.choice(["North","South","Central"]),
            "signup_date": faker.date_between(start_date="-2y", end_date="today"),
            "income": random.randint(300,2000)*1000
        })
    return pd.DataFrame(customers)

def generate_transactions(customers, max_tx=50):
    tx = []
    for _, row in customers.iterrows():
        n_tx = random.randint(1, max_tx)
        start_date = row["signup_date"]
        for i in range(n_tx):
            date = start_date + timedelta(days=random.randint(0,730))
            amount = round(random.uniform(5,200),2)
            tx.append({
                "customer_id": row["customer_id"],
                "transaction_date": date,
                "amount": amount,
                "channel": random.choice(["Online","Offline","App"])
            })
    return pd.DataFrame(tx)

if __name__ == "__main__":
    customers = generate_customers()
    transactions = generate_transactions(customers)

    customers.to_csv(r"C:\Users\phuoc\LTV_Optimization_Engine\data\raw\customers.csv", index=False)
    transactions.to_csv(r"C:\Users\phuoc\LTV_Optimization_Engine\data\raw\transactions.csv", index=False)

    print("✅ Generated mock datasets: customers.csv, transactions.csv")