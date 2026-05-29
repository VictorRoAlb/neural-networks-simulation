from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.utils import to_categorical


warnings.filterwarnings("ignore")


CLASS_NAMES = ["Low", "Medium", "High"]
CLASS_TO_INT = {"Low": 0, "Medium": 1, "High": 2}


@dataclass(slots=True)
class ExperimentConfig:
    data_path: Path
    output_dir: Path
    target_col: str = "Burnout_Risk"
    random_state: int = 42
    test_size: float = 0.10
    val_size_total: float = 0.10
    max_epochs: int = 50
    batch_size: int = 128
    patience: int = 10
    lr_patience: int = 5
    min_delta: float = 1e-4
    run_keras_tuner: bool = True
    save_outputs: bool = True


EXPERIMENTS = [
    {
        "model_name": "baseline_lineal_sgd",
        "description": "Linear softmax baseline",
        "hidden_layers": (),
        "activation": "relu",
        "dropout_rate": 0.0,
        "optimizer": "sgd",
        "learning_rate": 0.005,
    },
    {
        "model_name": "shallow_relu_sgd_lr_1e2",
        "description": "Shallow network, 200 ReLU units, SGD lr=0.01",
        "hidden_layers": (200,),
        "activation": "relu",
        "dropout_rate": 0.0,
        "optimizer": "sgd",
        "learning_rate": 0.01,
    },
    {
        "model_name": "shallow_relu_sgd_lr_1e3",
        "description": "Shallow network, 200 ReLU units, SGD lr=0.001",
        "hidden_layers": (200,),
        "activation": "relu",
        "dropout_rate": 0.0,
        "optimizer": "sgd",
        "learning_rate": 0.001,
    },
    {
        "model_name": "deep_relu_adam_no_dropout",
        "description": "Deep MLP 512-256-128-64-32, ReLU, Adam, no dropout",
        "hidden_layers": (512, 256, 128, 64, 32),
        "activation": "relu",
        "dropout_rate": 0.0,
        "optimizer": "adam",
        "learning_rate": 0.001,
    },
    {
        "model_name": "deep_relu_adam_dropout025",
        "description": "Deep MLP, ReLU, Adam, dropout=0.25",
        "hidden_layers": (512, 256, 128, 64, 32),
        "activation": "relu",
        "dropout_rate": 0.25,
        "optimizer": "adam",
        "learning_rate": 0.001,
    },
    {
        "model_name": "deep_relu_adam_dropout050",
        "description": "Deep MLP, ReLU, Adam, dropout=0.50",
        "hidden_layers": (512, 256, 128, 64, 32),
        "activation": "relu",
        "dropout_rate": 0.50,
        "optimizer": "adam",
        "learning_rate": 0.001,
    },
    {
        "model_name": "deep_tanh_adam_dropout025",
        "description": "Deep MLP, tanh, Adam, dropout=0.25",
        "hidden_layers": (512, 256, 128, 64, 32),
        "activation": "tanh",
        "dropout_rate": 0.25,
        "optimizer": "adam",
        "learning_rate": 0.001,
    },
    {
        "model_name": "deep_sigmoid_adam_dropout025",
        "description": "Deep MLP, sigmoid, Adam, dropout=0.25",
        "hidden_layers": (512, 256, 128, 64, 32),
        "activation": "sigmoid",
        "dropout_rate": 0.25,
        "optimizer": "adam",
        "learning_rate": 0.001,
    },
    {
        "model_name": "deep_elu_adam_dropout025",
        "description": "Deep MLP, ELU, Adam, dropout=0.25",
        "hidden_layers": (512, 256, 128, 64, 32),
        "activation": "elu",
        "dropout_rate": 0.25,
        "optimizer": "adam",
        "learning_rate": 0.001,
    },
    {
        "model_name": "deep_relu_sgd_dropout025",
        "description": "Deep MLP, ReLU, SGD, dropout=0.25",
        "hidden_layers": (512, 256, 128, 64, 32),
        "activation": "relu",
        "dropout_rate": 0.25,
        "optimizer": "sgd",
        "learning_rate": 0.001,
    },
    {
        "model_name": "deep_relu_rmsprop_dropout025",
        "description": "Deep MLP, ReLU, RMSprop, dropout=0.25",
        "hidden_layers": (512, 256, 128, 64, 32),
        "activation": "relu",
        "dropout_rate": 0.25,
        "optimizer": "rmsprop",
        "learning_rate": 0.001,
    },
]


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


class MacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        x_train: np.ndarray,
        y_train_int: np.ndarray,
        x_val: np.ndarray,
        y_val_int: np.ndarray,
        batch_size: int,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.x_train = x_train
        self.y_train_int = np.asarray(y_train_int)
        self.x_val = x_val
        self.y_val_int = np.asarray(y_val_int)
        self.batch_size = batch_size
        self.verbose = verbose

    def _compute_macro_f1(self, x_data: np.ndarray, y_true_int: np.ndarray) -> float:
        y_prob = self.model.predict(x_data, batch_size=self.batch_size, verbose=0)
        y_pred_int = np.argmax(y_prob, axis=1)
        return float(f1_score(y_true_int, y_pred_int, average="macro", zero_division=0))

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        train_macro_f1 = self._compute_macro_f1(self.x_train, self.y_train_int)
        val_macro_f1 = self._compute_macro_f1(self.x_val, self.y_val_int)
        logs["train_macro_f1"] = train_macro_f1
        logs["val_macro_f1"] = val_macro_f1
        if self.verbose:
            print(
                f"Epoch {epoch + 1}: train_macro_f1={train_macro_f1:.4f}, "
                f"val_macro_f1={val_macro_f1:.4f}"
            )


def get_optimizer(name: str, learning_rate: float):
    name = name.lower()
    if name == "sgd":
        return optimizers.SGD(learning_rate=learning_rate)
    if name == "adam":
        return optimizers.Adam(learning_rate=learning_rate)
    if name == "rmsprop":
        return optimizers.RMSprop(learning_rate=learning_rate)
    if name == "nadam":
        return optimizers.Nadam(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")


def build_mlp(
    input_dim: int,
    n_classes: int,
    hidden_layers: tuple[int, ...],
    activation: str,
    dropout_rate: float,
    optimizer_name: str,
    learning_rate: float,
    model_name: str,
):
    model = models.Sequential(name=model_name)
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(n_classes, activation="softmax"))
    model.compile(
        optimizer=get_optimizer(optimizer_name, learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_predictions(y_true_int: np.ndarray, y_pred_int: np.ndarray) -> tuple[dict, dict]:
    report = classification_report(
        y_true_int,
        y_pred_int,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    row = {
        "test_accuracy": accuracy_score(y_true_int, y_pred_int),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    for cls in CLASS_NAMES:
        row[f"{cls}_precision"] = report[cls]["precision"]
        row[f"{cls}_recall"] = report[cls]["recall"]
        row[f"{cls}_f1"] = report[cls]["f1-score"]
        row[f"{cls}_support"] = report[cls]["support"]
    return row, report


def load_and_split_data(config: ExperimentConfig):
    df = pd.read_excel(config.data_path)
    if config.target_col not in df.columns:
        raise ValueError(f"Target column '{config.target_col}' not found.")

    X = df.drop(columns=[config.target_col]).copy()
    y_raw = df[config.target_col].astype(str).str.strip().str.capitalize()
    y = y_raw.map(CLASS_TO_INT)

    if y.isna().any():
        unknown = sorted(y_raw[y.isna()].unique())
        raise ValueError(f"Unknown labels in {config.target_col}: {unknown}")

    y = y.astype(int)

    X_trainval, X_test, y_trainval, y_test_int = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    val_size_inside_trainval = config.val_size_total / (1.0 - config.test_size)
    X_train, X_val, y_train_int, y_val_int = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size_inside_trainval,
        stratify=y_trainval,
        random_state=config.random_state,
    )

    return df, X_train, X_val, X_test, y_train_int, y_val_int, y_test_int


def preprocess_splits(X_train, X_val, X_test, y_train_int, y_val_int, y_test_int):
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", encoder, cat_cols),
        ],
        remainder="drop",
    )

    x_train = preprocessor.fit_transform(X_train).astype("float32")
    x_val = preprocessor.transform(X_val).astype("float32")
    x_test = preprocessor.transform(X_test).astype("float32")

    y_train = to_categorical(y_train_int, num_classes=len(CLASS_NAMES))
    y_val = to_categorical(y_val_int, num_classes=len(CLASS_NAMES))
    y_test = to_categorical(y_test_int, num_classes=len(CLASS_NAMES))

    return preprocessor, x_train, x_val, x_test, y_train, y_val, y_test


def compute_class_weights(y_train_int: np.ndarray) -> tuple[dict[int, float], pd.DataFrame]:
    classes = np.unique(y_train_int)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_int)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    table = pd.DataFrame(
        {
            "class_name": [CLASS_NAMES[int(c)] for c in classes],
            "class_id": classes.astype(int),
            "n_train": [int((y_train_int == c).sum()) for c in classes],
            "train_percentage": [float((y_train_int == c).mean() * 100) for c in classes],
            "class_weight": weights,
        }
    )
    return class_weights, table


def make_callbacks(
    x_train: np.ndarray,
    y_train_int: np.ndarray,
    x_val: np.ndarray,
    y_val_int: np.ndarray,
    config: ExperimentConfig,
):
    return [
        MacroF1Callback(
            x_train=x_train,
            y_train_int=y_train_int,
            x_val=x_val,
            y_val_int=y_val_int,
            batch_size=config.batch_size,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=config.patience,
            min_delta=config.min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            mode="max",
            factor=0.5,
            patience=config.lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def fit_and_evaluate_model(
    experiment: dict,
    config: ExperimentConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_train_int: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_val_int: np.ndarray,
    x_test: np.ndarray,
    y_test_int: np.ndarray,
    class_weights: dict[int, float],
):
    set_global_seed(config.random_state)
    tf.keras.backend.clear_session()

    model = build_mlp(
        input_dim=x_train.shape[1],
        n_classes=len(CLASS_NAMES),
        hidden_layers=tuple(experiment.get("hidden_layers", ())),
        activation=experiment.get("activation", "relu"),
        dropout_rate=experiment.get("dropout_rate", 0.0),
        optimizer_name=experiment.get("optimizer", "adam"),
        learning_rate=experiment.get("learning_rate", 1e-3),
        model_name=experiment["model_name"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        class_weight=class_weights,
        callbacks=make_callbacks(x_train, y_train_int, x_val, y_val_int, config),
        verbose=1,
    )

    y_prob = model.predict(x_test, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    metrics_row, report = evaluate_predictions(y_test_int, y_pred)

    row = {
        **experiment,
        "epochs_trained": len(history.history["loss"]),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
        "best_train_macro_f1": float(np.max(history.history["train_macro_f1"])),
        "best_val_macro_f1": float(np.max(history.history["val_macro_f1"])),
        **metrics_row,
    }

    return model, history, row, report


def run_keras_tuner(
    config: ExperimentConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_train_int: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_val_int: np.ndarray,
    class_weights: dict[int, float],
):
    import keras_tuner as kt

    def build_tuner_model(hp):
        units = hp.Choice("units", values=[64, 128, 200, 256])
        activation = hp.Choice("activation", values=["relu", "tanh", "sigmoid"])
        learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        optimizer_name = hp.Choice("optimizer", values=["adam", "rmsprop"])
        return build_mlp(
            input_dim=x_train.shape[1],
            n_classes=len(CLASS_NAMES),
            hidden_layers=(units,),
            activation=activation,
            dropout_rate=0.0,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            model_name="keras_tuner_candidate",
        )

    tuner = kt.Hyperband(
        build_tuner_model,
        objective=kt.Objective("val_macro_f1", direction="max"),
        max_epochs=config.max_epochs,
        factor=3,
        seed=config.random_state,
        directory=str(config.output_dir / "keras_tuner_burnout_unified"),
        project_name="hyperband_clean_macro_f1",
        overwrite=True,
    )

    tuner.search(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        class_weight=class_weights,
        callbacks=make_callbacks(x_train, y_train_int, x_val, y_val_int, config),
        verbose=1,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return {
        "model_name": "keras_tuner_best_retrained",
        "description": "Best shallow network found by Keras Tuner and retrained under the common protocol",
        "hidden_layers": (best_hps.get("units"),),
        "activation": best_hps.get("activation"),
        "dropout_rate": 0.0,
        "optimizer": best_hps.get("optimizer"),
        "learning_rate": best_hps.get("learning_rate"),
    }


def save_learning_curve(history: dict, key_a: str, key_b: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history[key_a], label=key_a.replace("_", " "))
    ax.plot(history[key_b], label=key_b.replace("_", " "))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    config: ExperimentConfig,
    results_df: pd.DataFrame,
    final_table: pd.DataFrame,
    class_weight_table: pd.DataFrame,
    best_model,
    best_model_name: str,
    best_history,
    y_test_int: np.ndarray,
    y_pred_best: np.ndarray,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.output_dir / "summary_results.csv", index=False)
    final_table.to_csv(config.output_dir / "final_table_report.csv", index=False)
    class_weight_table.to_csv(config.output_dir / "class_weights.csv", index=False)
    best_model.save(config.output_dir / "best_model.keras")

    cm = confusion_matrix(y_test_int, y_pred_best, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Test confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    fig.savefig(config.output_dir / "confusion_matrix_best_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    best_hist = best_history.history
    save_learning_curve(
        best_hist,
        "accuracy",
        "val_accuracy",
        "Accuracy",
        "Best model - accuracy",
        config.output_dir / "curve_accuracy_best_model.png",
    )
    save_learning_curve(
        best_hist,
        "loss",
        "val_loss",
        "Loss",
        "Best model - loss",
        config.output_dir / "curve_loss_best_model.png",
    )
    if "train_macro_f1" in best_hist and "val_macro_f1" in best_hist:
        save_learning_curve(
            best_hist,
            "train_macro_f1",
            "val_macro_f1",
            "Macro F1",
            "Best model - Macro F1",
            config.output_dir / "curve_macro_f1_best_model.png",
        )
        save_learning_curve(
            best_hist,
            "train_macro_f1",
            "val_macro_f1",
            "Macro F1",
            "Best model - Macro F1",
            config.output_dir / "curve_val_macro_f1_best_model.png",
        )

    latex_cols = [
        "model_name",
        "best_train_macro_f1",
        "best_val_macro_f1",
        "test_accuracy",
        "macro_f1",
        "Low_f1",
        "epochs_trained",
    ]
    latex_str = results_df[latex_cols].to_latex(index=False, float_format="%.4f")
    (config.output_dir / "table_results_overleaf.tex").write_text(latex_str, encoding="utf-8")

    experiment_config = {
        "DATA_PATH": str(config.data_path),
        "TARGET_COL": config.target_col,
        "CLASS_NAMES": CLASS_NAMES,
        "CLASS_TO_INT": CLASS_TO_INT,
        "RANDOM_STATE": config.random_state,
        "TEST_SIZE": config.test_size,
        "VAL_SIZE_TOTAL": config.val_size_total,
        "MAX_EPOCHS": config.max_epochs,
        "BATCH_SIZE": config.batch_size,
        "PATIENCE": config.patience,
        "LR_PATIENCE": config.lr_patience,
        "MIN_DELTA": config.min_delta,
        "best_model_name": best_model_name,
        "selection_metric": "best_val_macro_f1",
        "monitor_metric_training": "val_macro_f1",
    }
    (config.output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run the burnout multiclass neural-network workflow.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the Excel dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_burnout_unified"),
        help="Directory where generated artifacts will be stored.",
    )
    parser.add_argument("--target-col", default="Burnout_Risk")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size-total", type=float, default=0.10)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--skip-keras-tuner", action="store_true")
    parser.add_argument("--no-save-outputs", action="store_true")
    args = parser.parse_args()
    return ExperimentConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        random_state=args.random_state,
        test_size=args.test_size,
        val_size_total=args.val_size_total,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr_patience=args.lr_patience,
        min_delta=args.min_delta,
        run_keras_tuner=not args.skip_keras_tuner,
        save_outputs=not args.no_save_outputs,
    )


def main() -> None:
    config = parse_args()
    set_global_seed(config.random_state)

    _, X_train, X_val, X_test, y_train_int, y_val_int, y_test_int = load_and_split_data(config)
    _, x_train, x_val, x_test, y_train, y_val, _ = preprocess_splits(
        X_train,
        X_val,
        X_test,
        y_train_int,
        y_val_int,
        y_test_int,
    )
    class_weights, class_weight_table = compute_class_weights(y_train_int)

    trained_models: dict[str, object] = {}
    histories: dict[str, object] = {}
    results: list[dict] = []

    for experiment in EXPERIMENTS:
        model, history, row, _ = fit_and_evaluate_model(
            experiment,
            config,
            x_train,
            y_train,
            y_train_int,
            x_val,
            y_val,
            y_val_int,
            x_test,
            y_test_int,
            class_weights,
        )
        trained_models[experiment["model_name"]] = model
        histories[experiment["model_name"]] = history
        results.append(row)

    if config.run_keras_tuner:
        tuner_experiment = run_keras_tuner(
            config,
            x_train,
            y_train,
            y_train_int,
            x_val,
            y_val,
            y_val_int,
            class_weights,
        )
        model, history, row, _ = fit_and_evaluate_model(
            tuner_experiment,
            config,
            x_train,
            y_train,
            y_train_int,
            x_val,
            y_val,
            y_val_int,
            x_test,
            y_test_int,
            class_weights,
        )
        trained_models[tuner_experiment["model_name"]] = model
        histories[tuner_experiment["model_name"]] = history
        results.append(row)

    results_df = pd.DataFrame(results).sort_values("best_val_macro_f1", ascending=False).reset_index(drop=True)
    cols_final = [
        "model_name",
        "description",
        "epochs_trained",
        "best_val_accuracy",
        "best_train_macro_f1",
        "best_val_macro_f1",
        "test_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "Low_precision",
        "Low_recall",
        "Low_f1",
        "Low_support",
        "Medium_f1",
        "High_f1",
    ]
    final_table = results_df[cols_final].copy()

    best_model_name = results_df.loc[0, "model_name"]
    best_model = trained_models[best_model_name]
    best_history = histories[best_model_name]

    y_prob_best = best_model.predict(x_test, verbose=0)
    y_pred_best = y_prob_best.argmax(axis=1)

    print("\nBest model:", best_model_name)
    print(results_df.loc[0, ["best_val_macro_f1", "macro_f1", "test_accuracy"]].to_string())

    if config.save_outputs:
        save_outputs(
            config,
            results_df,
            final_table,
            class_weight_table,
            best_model,
            best_model_name,
            best_history,
            np.asarray(y_test_int),
            y_pred_best,
        )


if __name__ == "__main__":
    main()
