import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_user_data(user_id, samples=100, genuine=True):
    data = []

    for _ in range(samples):
        if genuine:
            keystroke_mean = np.random.normal(180, 20)
            mouse_speed = np.random.normal(450, 50)
            click_duration = np.random.normal(120, 15)
            typing_pressure = np.random.normal(0.75, 0.08)
            error_rate = np.random.normal(0.04, 0.02)
            label = 1
        else:
            keystroke_mean = np.random.normal(260, 40)
            mouse_speed = np.random.normal(300, 80)
            click_duration = np.random.normal(180, 30)
            typing_pressure = np.random.normal(0.45, 0.12)
            error_rate = np.random.normal(0.15, 0.05)
            label = 0

        data.append([
            user_id,
            keystroke_mean,
            mouse_speed,
            click_duration,
            typing_pressure,
            error_rate,
            label
        ])

    return data


all_data = []

all_data += generate_user_data("user_1", 120, genuine=True)
all_data += generate_user_data("user_1", 120, genuine=False)

columns = [
    "user_id",
    "keystroke_mean",
    "mouse_speed",
    "click_duration",
    "typing_pressure",
    "error_rate",
    "label"
]

df = pd.DataFrame(all_data, columns=columns)

os.makedirs("sample_data", exist_ok=True)
df.to_csv("sample_data/behavioral_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())
