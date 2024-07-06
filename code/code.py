import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os

# Function to add numbers
def add_numbers(*args):
    return sum(args)

# Function to subtract numbers
def subtract_numbers(a, b):
    return a - b

# Function to calculate percentage
def calculate_percentage(part, whole):
    return (part / whole) * 100

# Function to read from a file and write to it
def read_and_write_file(read_path, write_path, content):
    with open(read_path, 'r') as file:
        data = file.read()
    
    with open(write_path, 'w') as file:
        file.write(data + "\n" + content)
    
    return "File written successfully."


# Function to create a simple GUI to login
def login_gui():
    def login():
        username = entry_username.get()
        password = entry_password.get()
        if username == "admin" and password == "password":
            messagebox.showinfo("Login", "Login successful")
        else:
            messagebox.showerror("Login", "Invalid credentials")

    root = tk.Tk()
    root.title("Login")
    
    tk.Label(root, text="Username").grid(row=0)
    tk.Label(root, text="Password").grid(row=1)
    
    entry_username = tk.Entry(root)
    entry_password = tk.Entry(root, show="*")
    
    entry_username.grid(row=0, column=1)
    entry_password.grid(row=1, column=1)
    
    tk.Button(root, text="Login", command=login).grid(row=2, column=1)
    
    root.mainloop()

# Function to register a user
def register_user(username, password):
    with open("users.txt", "a") as file:
        file.write(f"{username},{password}\n")
    return "User registered successfully."

# Example usage
if __name__ == "__main__":
    # Addition
    print(add_numbers(1, 2, 3, 4))
    
    # Subtraction
    print(subtract_numbers(10, 5))
    
    # Calculate percentage
    print(calculate_percentage(50, 200))
    
    # Read and write to a file
    print(read_and_write_file("readme.txt", "output.txt", "This is additional content."))
    
    # Calculate profit and loss
    print(calculate_profit_loss(100, 150))
    
    # Read Excel file
    print(read_excel_file("data.xlsx"))
    
    # GUI for login
    login_gui()
    
    # Register a user
    print(register_user("new_user", "new_password"))
