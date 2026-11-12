"""Compatibility entry point; prefer `python main.py train`."""
import json
import os

from dotenv import load_dotenv

from gold_ai.config import load_settings
from gold_ai.data_engine.pipeline import load_timeframe
from gold_ai.ml_engine import train_model

load_dotenv()


if __name__ == "__main__":
    settings = load_settings()
    timeframe = os.getenv("TRAIN_TIMEFRAME", "M15")
    summary_path = settings.path("processed_dir") / "build_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No processed dataset summary at {summary_path}; build real data first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("label") != "real":
        raise ValueError("Training requires data labeled 'real'; synthetic data is not accepted")
    metrics = train_model(
        load_timeframe(settings, timeframe),
        os.getenv("MODEL_PATH", settings["ml"]["model_path"]),
        int(os.getenv("PREDICTION_HORIZON", settings["ml"]["horizon_bars"])),
        float(os.getenv("LABEL_THRESHOLD", settings["ml"]["neutral_threshold"])),
        settings["backtest"]["train_frac"],
        settings["backtest"]["val_frac"],
        data_label="real",
    )
    print(metrics)
