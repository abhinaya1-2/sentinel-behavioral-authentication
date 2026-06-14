import joblib
import pandas as pd

model = joblib.load("sentinel_model.pkl")
scaler = joblib.load("scaler.pkl")

def authenticate_user(
    keystroke_mean,
    mouse_speed,
    click_duration,
    typing_pressure,
    error_rate
):
    input_data = pd.DataFrame(
        [[
            keystroke_mean,
            mouse_speed,
            click_duration,
            typing_pressure,
            error_rate
        ]],
        columns=[
            "keystroke_mean",
            "mouse_speed",
            "click_duration",
            "typing_pressure",
            "error_rate"
        ]
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        print("Authentication Result: Genuine User")
    else:
        print("Authentication Result: Impostor Detected")

    print("Confidence:", round(max(probability) * 100, 2), "%")


authenticate_user(
    keystroke_mean=175,
    mouse_speed=460,
    click_duration=118,
    typing_pressure=0.78,
    error_rate=0.03
)
