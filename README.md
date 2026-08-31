# 🏠 House Price Prediction — Streamlit App

A simple ML web app: enter house details (area, bedrooms, bathrooms, age, location score, etc.)
and get a predicted price from a Random Forest model, plus a feature-importance chart.

## Files
- `app.py` — the full app (data, model training, and UI in one file)
- `requirements.txt` — dependencies

## 1. Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

It'll open at `http://localhost:8501`.

## 2. Use your own data (optional)

Right now `app.py` generates synthetic data inside the `load_data()` function so the
app works out of the box with zero setup. To use a real dataset:

1. Put your CSV (with a `price` column) next to `app.py`, e.g. `housing.csv`.
2. Replace the body of `load_data()` with:
   ```python
   return pd.read_csv("housing.csv")
   ```
3. Make sure the other columns match what the UI collects (or edit the sliders in
   the "Enter house details" section to match your columns).

## 3. Host it for free on Streamlit Community Cloud

1. Create a GitHub repo and push `app.py` + `requirements.txt` to it (and your CSV, if used).
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **"New app"** → pick your repo, branch, and set the main file path to `app.py`.
4. Click **Deploy**. In a minute or two you'll get a public URL like
   `https://your-app-name.streamlit.app` that you can share with anyone.
5. Any time you push new commits to the repo, the hosted app auto-updates.

That's it — no server setup, no Docker, no config needed.
