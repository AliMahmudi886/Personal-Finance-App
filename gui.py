import tkinter as tk
from tkinter import messagebox
from models import User, Expense, Goal


class FinanceApp:
    def __init__(self, root):
        # Existing code...

        self.expense_category_label = tk.Label(root, text="Category")
        self.expense_category_label.pack()
        self.expense_category_entry = tk.Entry(root)
        self.expense_category_entry.pack()

        self.expense_amount_label = tk.Label(root, text="Amount")
        self.expense_amount_label.pack()
        self.expense_amount_entry = tk.Entry(root)
        self.expense_amount_entry.pack()

        self.expense_date_label = tk.Label(root, text="Date (YYYY-MM-DD)")
        self.expense_date_label.pack()
        self.expense_date_entry = tk.Entry(root)
        self.expense_date_entry.pack()

        self.log_expense_button = tk.Button(root, text="Log Expense", command=self.log_expense)
        self.log_expense_button.pack()

    def log_expense(self):
        user_id = 1  # Placeholder for the logged-in user ID
        category = self.expense_category_entry.get()
        amount = float(self.expense_amount_entry.get())
        date = self.expense_date_entry.get()
        expense = Expense(user_id, category, amount, date)
        expense.save()
        messagebox.showinfo("Success", "Expense logged successfully")


if __name__ == "__main__":
    root = tk.Tk()
    app = FinanceApp(root)
    root.mainloop()
