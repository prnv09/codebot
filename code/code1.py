import os 
import pandas as pd

# Function to calculate profit and loss
def calculate_profit_loss(cost_price, selling_price):
    if selling_price > cost_price:
        return f"Profit: {selling_price - cost_price}"
    elif selling_price < cost_price:
        return f"Loss: {cost_price - selling_price}"
    else:
        return "No Profit, No Loss"

# Function to read and print Excel file data
def read_excel_file(file_path):
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        return df.to_string()
    else:
        return "File not found."