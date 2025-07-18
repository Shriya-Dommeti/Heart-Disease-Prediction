from tkinter import *
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def show_entry_fields():
    try:
        # Get user inputs
        values = [
            int(e1.get()), int(e2.get()), int(e3.get()),
            int(e4.get()), int(e5.get()), int(e6.get())
        ]

        # Load model and supporting files
        model = joblib.load('model_joblib_heart')
        scaler = joblib.load('scaler_joblib_heart')
        feature_names = joblib.load('features_joblib_heart')

        # Load mean values for non-input features
        data = pd.read_csv('heart.csv')
        data = data.drop_duplicates()
        default_values = data[feature_names].mean()

        # Fill input values
        input_dict = default_values.to_dict()
        input_keys = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
        for key, val in zip(input_keys, values):
            input_dict[key] = val

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)

        # Predict and get probabilities
        result = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]

        # Show result text
        if result == 0:
            result_label.config(text="✅ No Heart Disease", fg="green")
        else:
            result_label.config(text="⚠️ Possibility of Heart Disease", fg="red")

        # Plot graph
        for widget in graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.bar(["No Disease", "Disease"], probs, color=['green', 'red'])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        for i, v in enumerate(probs):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    except Exception as e:
        result_label.config(text=f"Error: {e}", fg="red")

# Setup GUI
master = Tk()
master.title("Heart Disease Predictor")
master.geometry("500x600")
master.configure(bg="white")

Label(master, text="Heart Disease Prediction", font=("Helvetica", 16, 'bold'), bg="black", fg="white").pack(pady=10)

frame = Frame(master, bg="white")
frame.pack()

# Input fields
labels = [
    "Enter Age", "Sex (1=Male, 0=Female)", "Chest Pain Type (0-3)",
    "Resting BP", "Cholesterol", "Max Heart Rate"
]
entries = []

for i, text in enumerate(labels):
    Label(frame, text=text, font=("Arial", 10), bg="white").grid(row=i, column=0, pady=5, sticky=W)
    entry = Entry(frame)
    entry.grid(row=i, column=1, pady=5)
    entries.append(entry)

e1, e2, e3, e4, e5, e6 = entries

Button(master, text="Predict", command=show_entry_fields, bg="#007BFF", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

result_label = Label(master, text="", font=("Arial", 12), bg="white")
result_label.pack(pady=5)

graph_frame = Frame(master, bg="white")
graph_frame.pack(pady=10)

master.mainloop()
