# House Price Prediction — End-to-End ML Web App

An end-to-end machine learning product: a Jupyter notebook that cleans real Indian property
listing data and trains a price-prediction model, a FastAPI backend that serves it, and a
React + TypeScript frontend where a user enters property details and gets a predicted price.

## Overview

1. **`notebooks/`** — cleans the raw Kaggle dataset, explores it, trains and compares a Linear
   Regression baseline against a Random Forest Regressor, and exports the winning model as a
   single scikit-learn `Pipeline` (`house_price.pkl`) plus a `locations.json` file.
2. **`backend/`** — a FastAPI service that loads `house_price.pkl` once at startup and exposes
   `POST /predict`, `GET /health`, and `GET /locations`.
3. **`frontend/`** — a React + TypeScript + Vite single-page app with a form for property
   details and a result page showing the predicted price.

## Architecture

```
┌─────────────────┐        HTTP (JSON)        ┌──────────────────┐        joblib.load()      ┌────────────────────┐
│  React Frontend  │ ────────────────────────▶ │  FastAPI Backend  │ ─────────────────────────▶ │ house_price.pkl     │
│  (Vite, :5173)   │ ◀──────────────────────── │  (:8000)          │ ◀───────────────────────── │ (sklearn Pipeline)  │
└─────────────────┘   predicted_price (JSON)   └──────────────────┘      predicted price (log)  └────────────────────┘
                                                                                  ▲
                                                                                  │ trained & exported by
                                                                                  │
                                                                     ┌────────────────────────────┐
                                                                     │ notebooks/house_price_model │
                                                                     │ .ipynb (cleaning + training) │
                                                                     └────────────────────────────┘
```

## Tech Stack

| Layer      | Technology                                                        |
|------------|--------------------------------------------------------------------|
| Notebook   | Python, pandas, scikit-learn, matplotlib, seaborn                  |
| Backend    | FastAPI, Pydantic v2, pydantic-settings, uvicorn, joblib            |
| Frontend   | React 18, TypeScript, Vite, react-router-dom                       |
| Model      | scikit-learn `Pipeline` (`ColumnTransformer` + `RandomForestRegressor`) |

## Project Structure

```
.
├── notebooks/
│   ├── house_price_model.ipynb   # cleaning, EDA, training, evaluation, export
│   └── data/                     # place house_prices.csv here (gitignored)
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app, CORS, model loaded at startup (lifespan)
│   │   ├── api/routes/prediction.py   # GET /health, POST /predict, GET /locations
│   │   ├── core/config.py             # Settings from .env (pydantic-settings)
│   │   ├── schemas/prediction.py      # PredictionRequest / PredictionResponse
│   │   ├── services/
│   │   │   ├── preprocessing.py       # Turn a request into a one-row DataFrame
│   │   │   └── inference.py           # Load .pkl, run predict
│   │   └── utils/logging_config.py
│   ├── models/house_price.pkl         # ← copy from the notebook after training
│   ├── models/locations.json          # ← copy from the notebook after training
│   ├── tests/test_prediction.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/predictionClient.ts    # fetch wrapper, base URL from VITE_API_BASE_URL
│   │   ├── components/PredictionForm.tsx
│   │   ├── pages/HomePage.tsx | ResultPage.tsx | NotFoundPage.tsx
│   │   ├── types/prediction.ts        # TS types mirroring the backend schema
│   │   └── App.tsx                    # routes: / , /result , * (404)
│   ├── .env.example
│   └── package.json
├── .gitignore
└── README.md
```

## Dataset

**[House Price by Juhi Bhojani](https://www.kaggle.com/datasets/juhibhojani/house-price)** on
Kaggle — real property listings from India (`house_prices.csv`, ~187,000 rows).

### Download instructions

**Option A — manual:** click *Download* on the dataset page, unzip, and place the CSV at
`notebooks/data/house_prices.csv`.

**Option B — Kaggle CLI:**

```bash
pip install kaggle
# Get your API token: Kaggle → Settings → API → "Create New Token"
# Place kaggle.json in ~/.kaggle/ (macOS/Linux) or C:\Users\<you>\.kaggle\ (Windows)
kaggle datasets download -d juhibhojani/house-price -p notebooks/data --unzip
```

## Setup & Running the Notebook

```bash
cd notebooks
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install jupyter pandas numpy scikit-learn matplotlib seaborn joblib
jupyter notebook house_price_model.ipynb
```

Run all cells top-to-bottom (Kernel → Restart & Run All). This produces `house_price.pkl` and
`locations.json` in the `notebooks/` folder — copy both into `backend/models/`.

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# make sure models/house_price.pkl and models/locations.json exist (see above)

uvicorn app.main:app --reload
# open http://localhost:8000/docs to try /predict from the Swagger UI
```

Run the tests:

```bash
pytest
```

### Environment variables (`backend/.env`)

| Variable          | Description                                | Default                     |
|-------------------|---------------------------------------------|------------------------------|
| `APP_NAME`        | Display name of the API                     | `House Price Prediction API` |
| `CORS_ORIGINS`     | Comma-separated allowed origins              | `http://localhost:5173`      |
| `MODEL_PATH`       | Path to the exported `.pkl` pipeline         | `models/house_price.pkl`     |
| `LOCATIONS_PATH`   | Path to the exported `locations.json`        | `models/locations.json`      |

> ⚠️ **Version pinning:** a pickle only loads reliably with the same scikit-learn version it was
> trained with. The notebook prints `sklearn.__version__` right before exporting the model — pin
> that exact version in `backend/requirements.txt`.

## Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
# open http://localhost:5173
```

### Environment variables (`frontend/.env`)

| Variable               | Description                     | Default                  |
|-------------------------|----------------------------------|---------------------------|
| `VITE_API_BASE_URL`     | Base URL of the FastAPI backend  | `http://localhost:8000`   |

## API Reference

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /locations`

Returns the list of location names the model was trained on (used to populate the frontend
dropdown). Unknown locations sent to `/predict` are automatically mapped to `"other"`.

### `POST /predict`

Request body:

```json
{
  "location": "thane",
  "carpet_area_sqft": 650,
  "floor_num": 4,
  "bathroom": 2,
  "balcony": 1,
  "furnishing": "Semi-Furnished",
  "transaction": "Resale",
  "ownership": "Freehold",
  "facing": "East"
}
```

Response:

```json
{ "predicted_price": 6750000.0 }
```

Example with `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "thane",
    "carpet_area_sqft": 650,
    "floor_num": 4,
    "bathroom": 2,
    "balcony": 1,
    "furnishing": "Semi-Furnished",
    "transaction": "Resale",
    "ownership": "Freehold",
    "facing": "East"
  }'
```

## Model Metrics

The notebook trains and compares two models on a held-out 20% test set (never on training data).
**Fill in the exact numbers printed by the notebook's `comparison_df` after you run it on your
downloaded copy of the dataset** — they'll vary slightly by dataset version and random split:

| Model              | MAE (₹) | RMSE (₹) | R²   |
|---------------------|---------|----------|------|
| Linear Regression    | *(see notebook output)* | *(see notebook output)* | *(see notebook output)* |
| Random Forest (final)| *(see notebook output)* | *(see notebook output)* | *(see notebook output)* |

Random Forest is selected as the final exported model — it consistently captures the non-linear
interactions between location, area, and floor better than a plain linear model on this dataset.

## Screenshots

*(Add screenshots of the running form and result page here after you run the app end-to-end.)*

## Verifying the Full Flow

1. Backend running on `:8000` (`uvicorn app.main:app --reload`)
2. Frontend running on `:5173` (`npm run dev`)
3. Open `http://localhost:5173`, fill in the form, submit, and confirm a real predicted price
   appears on the result page.
