import gradio as gr
import joblib
import pandas as pd
import xgboost as xgb

# Model ve scaler yükleme
model = xgb.Booster()
model.load_model("xgb_model.json")
scaler = joblib.load("scaler.pkl")

# Tahmin fonksiyonu
def predict(pressure, temp_x_pressure, fusion_metric):
    input_df = pd.DataFrame([[pressure, temp_x_pressure, fusion_metric]],
                            columns=["Pressure (kPa)", "Temperature x Pressure", "Material Fusion Metric"])
    scaled = scaler.transform(input_df)
    dmatrix = xgb.DMatrix(scaled)
    prediction = model.predict(dmatrix)[0]
    return float(prediction)

# Gradio arayüzü
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Pressure (kPa)"),
        gr.Number(label="Temperature x Pressure"),
        gr.Number(label="Material Fusion Metric")
    ],
    outputs=gr.Number(label="Kalite Skoru"),
    title="Kalite Skoru Tahmin Modeli",
    description="Pressure, Temperature x Pressure ve Material Fusion Metric değerlerini giriniz, kalite skorunu tahmin eder.",
)

iface.queue().launch()  # Hugging Face için önerilen biçim




