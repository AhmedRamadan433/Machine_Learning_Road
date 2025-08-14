import tkinter as tk
from tkinter import filedialog, ttk, messagebox, Toplevel, Checkbutton, IntVar, Canvas, Scrollbar, Frame
import pandas as pd
import numpy as np
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error, silhouette_score, davies_bouldin_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input
from keras.optimizers import Adam
import warnings
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# ================= CONFIGURATION DATA=================
@dataclass
class ModelParam:
    """Data class for model parameters."""
    name: str
    label: str
    type: str
    default: str
    min: Optional[float] = None
    options: Optional[list] = None

class Config:
    """Configuration class holding model options, metrics, parameters, scalers, and null handling options."""
    MODEL_OPTIONS = {
        "Classification": ["KNN", "SVM", "Logistic Regression"],
        "Regression": ["Linear Regression"],
        "Clustering": ["K-Means"],
        "Deep": ["ANN"]
    }

    METRIC_OPTIONS = {
        "Classification": ["Accuracy", "Precision", "Recall", "F1", "Confusion Matrix"],
        "Regression": ["R2", "MSE", "MAE", "RMSE"],
        "Clustering": ["Silhouette", "Davies-Bouldin"],
        "Deep": ["Accuracy", "Loss"]
    }

    MODEL_PARAMS = {
        "KNN": [ModelParam("n_neighbors", "Number of Neighbors", "int", "5", 1)],
        "SVM": [
            ModelParam("C", "Regularization Parameter (C)", "float", "1.0", 0.01),
            ModelParam("kernel", "Kernel", "choice", "rbf", options=["linear", "rbf", "poly"])
        ],
        "Logistic Regression": [
            ModelParam("max_iter", "Maximum Iterations", "int", "2000", 100),
            ModelParam("C", "Regularization Strength (C)", "float", "1.0", 0.01),
            ModelParam("solver", "Solver", "choice", "lbfgs", options=["lbfgs", "liblinear", "saga"])
        ],
        "Linear Regression": [],
        "K-Means": [ModelParam("n_clusters", "Number of Clusters", "int", "3", 2)],
        "ANN": [
            ModelParam("hidden_units", "Hidden Units", "int", "256", 1),
            ModelParam("num_hidden_layers", "Number of Hidden Layers", "int", "2", 1),
            ModelParam("dropout", "Dropout Rate", "float", "0.4", 0.0),
            ModelParam("learning_rate", "Learning Rate", "float", "0.001", 0.0001),
            ModelParam("activation", "Activation Function", "choice", "relu", options=["relu", "tanh", "sigmoid"]),
            ModelParam("epochs", "Epochs", "int", "20", 1),
            ModelParam("batch_size", "Batch Size", "int", "128", 1)
        ]
    }

    SCALER_OPTIONS = ["StandardScaler", "MinMaxScaler"]

    NULL_HANDLING_OPTIONS = [
        "Drop Rows with Nulls", "Drop Columns with Nulls",
        "Fill Nulls (mean)", "Fill Nulls (median)", "Fill Nulls (mode)",
        "Remove Duplicates"
    ]

# ================= DATA MODELS =================
class Dataset:
    """Class to manage dataset loading, preprocessing, and information."""
    def __init__(self, data: pd.DataFrame = None, scaler_type: str = "StandardScaler"):
        self._data = data
        self._scaler_type = scaler_type
        self._scaler = StandardScaler() if scaler_type == "StandardScaler" else MinMaxScaler()

    @property
    def data(self) -> Optional[pd.DataFrame]:
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame):
        self._data = value
        self._scaler = StandardScaler() if self._scaler_type == "StandardScaler" else MinMaxScaler()

    @property
    def columns(self) -> list:
        return list(self._data.columns) if self._data is not None else []

    @property
    def shape(self) -> tuple:
        return self._data.shape if self._data is not None else (0, 0)

    @property
    def scaler_type(self) -> str:
        return self._scaler_type

    @scaler_type.setter
    def scaler_type(self, value: str):
        if value not in Config.SCALER_OPTIONS:
            raise ValueError(f"Invalid scaler type: {value}. Choose from {Config.SCALER_OPTIONS}")
        self._scaler_type = value
        self._scaler = StandardScaler() if value == "StandardScaler" else MinMaxScaler()

    def is_loaded(self) -> bool:
        """Check if a dataset is loaded and not empty."""
        return self._data is not None and not self._data.empty

    def get_info(self) -> str:
        """Get dataset information including shape, nulls, and counts."""
        if not self.is_loaded():
            return "No dataset loaded"

        nulls = self._data.isnull().sum()
        counts = self._data.count()
        info_lines = [
            "Dataset Info:\n",
            f"Shape: {self._data.shape}\n",
            f"Scaler: {self._scaler_type}\n"
        ]

        for col in self._data.columns:
            info_lines.append(f"{col}: Nulls={nulls[col]}, Count={counts[col]}, Dtype={self._data[col].dtype}")

        return "\n".join(info_lines)

    def handle_nulls(self, method: str) -> Tuple[bool, str]:
        """Handle null values and duplicates in the dataset based on the selected method."""
        if not self.is_loaded():
            return False, "No dataset loaded"

        try:
            before_shape = self._data.shape
            if method == "Drop Rows with Nulls":
                self._data = self._data.dropna()
                return True, f"Dropped rows with null values. Shape: {before_shape} -> {self._data.shape}"

            elif method == "Drop Columns with Nulls":
                before_columns = list(self._data.columns)
                self._data = self._data.dropna(axis=1)
                after_columns = list(self._data.columns)
                dropped_columns = [col for col in before_columns if col not in after_columns]
                return True, f"Dropped columns with null values: {dropped_columns}. Shape: {before_shape} -> {self._data.shape}"

            elif method == "Fill Nulls (mean)":
                num_cols = self._data.select_dtypes(include=[np.number]).columns
                before_na = self._data[num_cols].isna().sum().sum()
                self._data[num_cols] = self._data[num_cols].fillna(self._data[num_cols].mean(numeric_only=True))
                after_na = self._data[num_cols].isna().sum().sum()
                return True, f"Filled numeric nulls with mean. NaNs in numeric: {before_na} -> {after_na}"

            elif method == "Fill Nulls (median)":
                num_cols = self._data.select_dtypes(include=[np.number]).columns
                before_na = self._data[num_cols].isna().sum().sum()
                self._data[num_cols] = self._data[num_cols].fillna(self._data[num_cols].median(numeric_only=True))
                after_na = self._data[num_cols].isna().sum().sum()
                return True, f"Filled numeric nulls with median. NaNs in numeric: {before_na} -> {after_na}"

            elif method == "Fill Nulls (mode)":
                before_na = self._data.isna().sum().sum()
                for col in self._data.columns:
                    if self._data[col].isna().any():
                        mode_val = self._data[col].mode()[0] if not self._data[col].mode().empty else np.nan
                        self._data[col] = self._data[col].fillna(mode_val)
                after_na = self._data.isna().sum().sum()
                return True, f"Filled nulls with mode (per column). Total NaNs: {before_na} -> {after_na}"

            elif method == "Remove Duplicates":
                self._data = self._data.drop_duplicates()
                return True, f"Removed duplicate rows. Shape: {before_shape} -> {self._data.shape}"

        except Exception as e:
            logging.error(f"Error handling nulls: {str(e)}")
            return False, f"Error handling nulls: {str(e)}"

        return False, f"Unknown null handling method: {method}"

    def drop_columns(self, columns: List[str]) -> Tuple[bool, str]:
        """Drop specified columns from the dataset."""
        if not self.is_loaded():
            return False, "No dataset loaded"

        try:
            before_shape = self._data.shape
            valid_columns = [col for col in columns if col in self._data.columns]
            if not valid_columns:
                return False, "No valid columns selected for dropping"
            self._data = self._data.drop(columns=valid_columns)
            logging.info(f"Dropped columns: {valid_columns}")
            return True, f"Dropped columns: {valid_columns}. Shape: {before_shape} -> {self._data.shape}"
        except Exception as e:
            logging.error(f"Error dropping columns: {str(e)}")
            return False, f"Error dropping columns: {str(e)}"

    def prepare_features(self, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]:
        """Prepare features by encoding categoricals and scaling numerics."""
        if not self.is_loaded():
            raise ValueError("No dataset loaded")

        df = self._data.copy() if target_col is None else self._data.drop(columns=[target_col])
        df_encoded = pd.get_dummies(df, drop_first=False, dtype=float)
        num_cols = df_encoded.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df_encoded[num_cols] = self._scaler.fit_transform(df_encoded[num_cols])
        return df_encoded, self._scaler

# ================= MODEL FACTORIES =================
class ModelFactory:
    """Factory class to create machine learning models based on name and parameters."""
    @staticmethod
    def create_model(model_name: str, params: Dict[str, Any]):
        model_creators = {
            "KNN": lambda p: KNeighborsClassifier(n_neighbors=p.get("n_neighbors", 5)),
            "SVM": lambda p: SVC(C=p.get("C", 1.0), kernel=p.get("kernel", "rbf"), probability=True),
            "Logistic Regression": lambda p: LogisticRegression(
                max_iter=p.get("max_iter", 2000),
                C=p.get("C", 1.0),
                solver=p.get("solver", "lbfgs")
            ),
            "Linear Regression": lambda p: LinearRegression(),
            "K-Means": lambda p: KMeans(n_clusters=p.get("n_clusters", 3), n_init='auto', random_state=42),
            "ANN": lambda p: Sequential([
                Input(shape=(p.get("input_size"),)),
                *[layer for i in range(p.get("num_hidden_layers", 2))
                  for layer in [Dense(p.get("hidden_units", 256)),
                                Activation(p.get("activation", "relu")),
                                Dropout(p.get("dropout", 0.4))]],
                Dense(p.get("num_labels"), activation='softmax')
            ])
        }

        if model_name not in model_creators:
            raise ValueError(f"Unknown model: {model_name}")

        model = model_creators[model_name](params)
        if model_name == "ANN":
            model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=params.get("learning_rate", 0.001)),
                metrics=['accuracy']
            )
        return model

# ================= MODEL RUNNERS =================
class ModelRunner(ABC):
    """Abstract base class for model runners."""
    @abstractmethod
    def run(self, model, X, y=None, metric: str = None, **kwargs) -> Dict[str, Any]:
        pass

class ClassificationRunner(ModelRunner):
    """Runner for classification models."""
    def run(self, model, X_train, X_test, y_train, y_test, metric: str, class_map=None) -> Dict[str, Any]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)
            for warning in w:
                if "ConvergenceWarning" in str(warning.message):
                    logging.warning(f"Model convergence warning: {str(warning.message)}")
                    messagebox.showwarning(
                        "Convergence Warning",
                        f"Model failed to converge: {str(warning.message)}\n"
                        "Try increasing max_iter, changing solver, or ensuring data is scaled."
                    )
            y_pred = model.predict(X_test)

        metrics = {
            "Accuracy": lambda: {"Accuracy": round(accuracy_score(y_test, y_pred), 4)},
            "Precision": lambda: {
                "Precision (weighted)": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4)},
            "Recall": lambda: {
                "Recall (weighted)": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4)},
            "F1": lambda: {
                "F1 Score (weighted)": round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)},
            "Confusion Matrix": lambda: {
                "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
                **({"Label Mapping": class_map} if class_map else {})
            }
        }

        return metrics.get(metric, lambda: {})()

class RegressionRunner(ModelRunner):
    """Runner for regression models."""
    def run(self, model, X_train, X_test, y_train, y_test, metric: str) -> Dict[str, Any]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "R2": lambda: {"R² Score": round(r2_score(y_test, y_pred), 4)},
            "MSE": lambda: {"Mean Squared Error": round(mean_squared_error(y_test, y_pred), 4)},
            "MAE": lambda: {"Mean Absolute Error": round(mean_absolute_error(y_test, y_pred), 4)},
            "RMSE": lambda: {"Root Mean Squared Error": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)}
        }

        return metrics.get(metric, lambda: {})()

class ClusteringRunner(ModelRunner):
    """Runner for clustering models."""
    def run(self, model, X, metric: str) -> Dict[str, Any]:
        y_pred = model.fit_predict(X)

        def compute_silhouette():
            try:
                if len(np.unique(y_pred)) > 1 and len(y_pred) > len(np.unique(y_pred)):
                    return {"Silhouette Score": round(silhouette_score(X, y_pred), 4)}
                else:
                    return {"Silhouette Score": "Not computed: Insufficient clusters or samples"}
            except Exception as e:
                logging.error(f"Silhouette score computation error: {str(e)}")
                return {"Silhouette Score": f"Not computed: {str(e)}"}

        def compute_davies_bouldin():
            try:
                return {"Davies-Bouldin Index": round(davies_bouldin_score(X, y_pred), 4)}
            except Exception as e:
                logging.error(f"Davies-Bouldin score computation error: {str(e)}")
                return {"Davies-Bouldin Index": f"Not computed: {str(e)}"}

        metrics = {
            "Silhouette": compute_silhouette,
            "Davies-Bouldin": compute_davies_bouldin
        }

        return metrics.get(metric, lambda: {})()

class ANNRunner(ModelRunner):
    """Runner for ANN models."""
    def run(self, model, X_train, X_test, y_train, y_test, metric: str, class_map=None, epochs=20, batch_size=128) -> Dict[str, Any]:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        metrics = {
            "Accuracy": lambda: {"Accuracy": round(accuracy, 4)},
            "Loss": lambda: {"Loss": round(loss, 4)},
            "Precision": lambda: {
                "Precision (weighted)": round(precision_score(y_test_classes, y_pred, average='weighted', zero_division=0), 4)},
            "Recall": lambda: {
                "Recall (weighted)": round(recall_score(y_test_classes, y_pred, average='weighted', zero_division=0), 4)},
            "F1": lambda: {
                "F1 Score (weighted)": round(f1_score(y_test_classes, y_pred, average='weighted', zero_division=0), 4)},
            "Confusion Matrix": lambda: {
                "Confusion Matrix": confusion_matrix(y_test_classes, y_pred).tolist(),
                **({"Label Mapping": class_map} if class_map else {})
            }
        }

        return metrics.get(metric, lambda: {})()

# ================= VALIDATORS =================
class Validator:
    """Class for validating user inputs."""
    @staticmethod
    def validate_test_size(test_size_str: str) -> float:
        try:
            test_size = float(test_size_str)
            if not 0.0 < test_size < 1.0:
                raise ValueError("Test size must be between 0 and 1 (exclusive).")
            return test_size
        except ValueError:
            raise ValueError(f"Invalid test size: {test_size_str}. Must be a float between 0 and 1.")

    @staticmethod
    def validate_param(param: ModelParam, value: str) -> Any:
        try:
            if param.type == "int":
                val = int(value)
                if param.min is not None and val < param.min:
                    raise ValueError(f"{param.label} must be at least {param.min}")
                return val
            elif param.type == "float":
                val = float(value)
                if param.min is not None and val < param.min:
                    raise ValueError(f"{param.label} must be at least {param.min}")
                return val
            elif param.type == "choice":
                if param.options and value not in param.options:
                    raise ValueError(f"{param.label} must be one of {param.options}")
                return value
        except ValueError as e:
            raise ValueError(f"Invalid {param.label}: {str(e)}")

# ================= MAIN APPLICATION =================
class MLApp:
    """Main application"""
    def __init__(self):
        self.dataset = Dataset()
        self.current_params = {}
        self.param_entries = {}
        self.runners = {
            "Classification": ClassificationRunner(),
            "Regression": RegressionRunner(),
            "Clustering": ClusteringRunner(),
            "Deep": ANNRunner()
        }

        self.setup_gui()

    def setup_gui(self):
        """Set up the main GUI window and widgets."""
        self.root = tk.Tk()
        self.root.title("Final Project")
        self.root.geometry("900x700")
        self.create_widgets()
        self.initialize_options()

    def create_widgets(self):
        """Create all GUI widgets and frames."""
        self.create_data_frame()
        self.create_model_selection_frames()
        self.create_parameters_frame()
        self.create_target_nulls_frame()
        self.create_split_frame()
        self.create_metrics_frame()
        self.create_run_frame()

    def create_data_frame(self):
        """Create frame for data loading and info."""
        frm1 = tk.LabelFrame(self.root, text="1) Load Data", padx=8, pady=8)
        frm1.pack(fill="x", padx=10, pady=6)

        tk.Button(frm1, text="Load CSV", command=self.load_csv).pack(side="left", padx=4)
        tk.Button(frm1, text="Show Data Info", command=self.show_data_info).pack(side="left", padx=4)

    def create_model_selection_frames(self):
        """Create frames for problem type and model selection."""
        frm2a = tk.LabelFrame(self.root, text="2) Problem Type", padx=8, pady=8)
        frm2a.pack(fill="x", padx=10, pady=6)

        self.problem_choice = ttk.Combobox(frm2a, values=list(Config.MODEL_OPTIONS.keys()),
                                           state="readonly", width=30)
        self.problem_choice.set("Classification")
        self.problem_choice.pack(side="left", padx=4)
        self.problem_choice.bind("<<ComboboxSelected>>", self.update_model_and_metric_options)

        frm2b = tk.LabelFrame(self.root, text="3) Model Choice", padx=8, pady=8)
        frm2b.pack(fill="x", padx=10, pady=6)

        self.model_choice = ttk.Combobox(frm2b, state="readonly", width=30)
        self.model_choice.pack(side="left", padx=4)
        self.model_choice.bind("<<ComboboxSelected>>", self.on_model_change)

    def create_parameters_frame(self):
        """Create frame for model parameters with scrollable canvas."""
        frm2c = tk.LabelFrame(self.root, text="4) Model Parameters", padx=8, pady=8)
        frm2c.pack(fill="x", padx=10, pady=6)

        # Create canvas with scrollbar
        self.canvas = Canvas(frm2c, height=150)  # Fixed height for the frame
        scrollbar = Scrollbar(frm2c, orient="vertical", command=self.canvas.yview)
        self.param_frame = Frame(self.canvas)

        # Configure canvas and scrollbar
        self.param_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.param_frame, anchor="nw")

        # Handle window resizing
        def on_canvas_configure(event):
            self.canvas.itemconfig(self.canvas_frame, width=event.width)

        self.canvas.bind("<Configure>", on_canvas_configure)

        # Add mouse wheel scrolling
        def _on_mousewheel(event):
            if event.delta:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        self.root.bind_all("<Button-4>", _on_mousewheel)
        self.root.bind_all("<Button-5>", _on_mousewheel)

    def create_target_nulls_frame(self):
        """Create frame for target selection, null handling, and scaler."""
        frm3 = tk.LabelFrame(self.root, text="5) Select Target, Nulls and Scaler", padx=8, pady=8)
        frm3.pack(fill="x", padx=10, pady=6)

        self.target_label = tk.Label(frm3, text="Target:")
        self.target_label.pack(side="left")

        self.target_menu = ttk.Combobox(frm3, width=30, state="readonly")
        self.target_menu.pack(side="left", padx=6)

        tk.Label(frm3, text="Nulls:").pack(side="left", padx=(12, 0))

        self.null_method = ttk.Combobox(frm3, values=Config.NULL_HANDLING_OPTIONS,
                                        state="readonly", width=20)
        self.null_method.set("Fill Nulls (mean)")
        self.null_method.pack(side="left", padx=6)

        tk.Label(frm3, text="Scaler:").pack(side="left", padx=(12, 0))

        self.scaler_choice = ttk.Combobox(frm3, values=Config.SCALER_OPTIONS,
                                          state="readonly", width=20)
        self.scaler_choice.set("StandardScaler")
        self.scaler_choice.pack(side="left", padx=6)
        self.scaler_choice.bind("<<ComboboxSelected>>", self.update_scaler)

        tk.Button(frm3, text="Apply", command=self.handle_nulls).pack(side="left", padx=6)
        tk.Button(frm3, text="Drop Columns", command=self.drop_columns_dialog).pack(side="left", padx=6)

    def create_split_frame(self):
        """Create frame for train/test split."""
        frm4 = tk.LabelFrame(self.root, text="6) Train/Test Split", padx=8, pady=8)
        frm4.pack(fill="x", padx=10, pady=6)

        self.test_size_label = tk.Label(frm4, text="Test size (0-1):")
        self.test_size_label.pack(side="left")

        self.test_size_entry = tk.Entry(frm4, width=8)
        self.test_size_entry.insert(0, "0.2")
        self.test_size_entry.pack(side="left", padx=6)

    def create_metrics_frame(self):
        """Create frame for metric selection."""
        frm5 = tk.LabelFrame(self.root, text="7) Select Metric", padx=8, pady=8)
        frm5.pack(fill="x", padx=10, pady=6)

        self.metric_choice = ttk.Combobox(frm5, state="readonly", width=30)
        self.metric_choice.pack(side="left", padx=4)

    def create_run_frame(self):
        """Create frame for run and exit buttons."""
        frm_run = tk.Frame(self.root, padx=8, pady=4)
        frm_run.pack(fill="x", padx=10, pady=10)

        tk.Button(frm_run, text="Run Model", command=self.run_model,
                  bg="#4CAF50", fg="white", font=("Arial", 11, "bold")).pack(side="left", padx=(0, 10))

        tk.Button(frm_run, text="Exit", command=self.exit_application,
                  bg="#f44336", fg="white", font=("Arial", 11, "bold")).pack(side="left")

    def load_csv(self) -> None:
        """Load a CSV file into the dataset."""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not file_path:
            return

        try:
            data = pd.read_csv(file_path)
            self.dataset.data = data
            self.target_menu['values'] = self.dataset.columns
            self.target_menu.set('')
            logging.info(f"Dataset loaded: {file_path}, Shape: {self.dataset.shape}")
            messagebox.showinfo("Data Loaded", f"Loaded:\nShape: {self.dataset.shape}")
        except Exception as e:
            logging.error(f"Failed to load CSV: {str(e)}")
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")

    def show_data_info(self) -> None:
        """Show dataset information in a message box."""
        if not self.dataset.is_loaded():
            messagebox.showerror("Error", "Load a dataset first.")
            return

        messagebox.showinfo("Dataset Info", self.dataset.get_info())

    def handle_nulls(self) -> None:
        """Apply null handling method to the dataset."""
        success, message = self.dataset.handle_nulls(self.null_method.get())
        if success:
            logging.info(message)
            messagebox.showinfo("Null Handling", message)
            self.target_menu['values'] = self.dataset.columns
            self.target_menu.set('')
        else:
            messagebox.showerror("Error", message)

    def drop_columns_dialog(self) -> None:
        """Open a dialog to select columns to drop"""
        if not self.dataset.is_loaded():
            messagebox.showerror("Error", "Load a dataset first.")
            return

        dialog = Toplevel(self.root)
        dialog.title("Select Columns to Drop")

        num_columns = len(self.dataset.columns)
        max_col_length = max((len(col) for col in self.dataset.columns), default=10)
        char_width = 8
        window_width = min(max_col_length * char_width + 150, 600)
        window_height = min(num_columns * 30 + 100, 500)
        dialog.geometry(f"{int(window_width)}x{int(window_height)}")

        canvas = Canvas(dialog)
        scrollbar = Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Instruction label
        tk.Label(scrollable_frame, text="Select columns to drop:", font=("Arial", 10)).pack(pady=5, anchor="w", padx=10)

        # Dictionary to store checkbox states
        column_vars = {}
        for col in self.dataset.columns:
            var = IntVar()
            column_vars[col] = var
            frame = Frame(scrollable_frame)
            frame.pack(fill="x", padx=10, pady=2)
            Checkbutton(frame, text=col, variable=var, anchor="w", font=("Arial", 9)).pack(side="left")

        def confirm_drop():
            columns_to_drop = [col for col, var in column_vars.items() if var.get() == 1]
            if not columns_to_drop:
                messagebox.showinfo("No Selection", "No columns selected to drop.")
                dialog.destroy()
                return

            success, message = self.dataset.drop_columns(columns_to_drop)
            if success:
                logging.info(message)
                messagebox.showinfo("Success", message)
                self.target_menu['values'] = self.dataset.columns
                self.target_menu.set('')
            else:
                messagebox.showerror("Error", message)
            dialog.destroy()

        # Buttons frame
        btn_frame = Frame(scrollable_frame)
        btn_frame.pack(pady=10, fill="x")
        tk.Button(btn_frame, text="Confirm", command=confirm_drop, font=("Arial", 10)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, font=("Arial", 10)).pack(side="left", padx=5)

        # Handle window resizing
        def on_frame_configure(event):
            canvas.itemconfig(canvas_frame, width=event.width)

        canvas.bind("<Configure>", on_frame_configure)

        # Add mouse wheel scrolling
        def _on_mousewheel(event):
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        dialog.bind_all("<MouseWheel>", _on_mousewheel)
        dialog.bind_all("<Button-4>", _on_mousewheel)
        dialog.bind_all("<Button-5>", _on_mousewheel)

    def update_scaler(self, event=None) -> None:
        """Update the scaler type in the dataset."""
        try:
            self.dataset.scaler_type = self.scaler_choice.get()
            logging.info(f"Scaler updated to: {self.dataset.scaler_type}")
            messagebox.showinfo("Scaler Updated", f"Scaler set to: {self.dataset.scaler_type}")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def update_parameters(self, model_name: str) -> None:
        """Update model parameters from inputs."""
        params = {}
        param_configs = Config.MODEL_PARAMS.get(model_name, [])

        for param_config in param_configs:
            entry = self.param_entries.get(param_config.name)
            if entry:
                try:
                    value = entry.get()
                    params[param_config.name] = Validator.validate_param(param_config, value)
                except ValueError as e:
                    messagebox.showerror("Validation Error", str(e))
                    return

        self.current_params[model_name] = params
        logging.info(f"Parameters updated for {model_name}: {params}")
        messagebox.showinfo("Parameters Updated",
                            f"Parameters for {model_name} updated:\n{params}")

    def update_param_inputs(self, model_name: str) -> None:
        """Update parameter input fields based on selected model."""
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()

        param_configs = Config.MODEL_PARAMS.get(model_name, [])
        if not param_configs:
            tk.Label(self.param_frame, text="No parameters to configure.").pack(pady=5, padx=10)
            return

        for idx, config in enumerate(param_configs):
            frame = Frame(self.param_frame)
            frame.pack(fill="x", padx=10, pady=5)

            tk.Label(frame, text=config.label + ":", font=("Arial", 9)).pack(side="left", padx=4)

            if config.type == "choice":
                entry = ttk.Combobox(frame, values=config.options, state="readonly", width=15)
                entry.set(config.default)
            else:
                entry = tk.Entry(frame, width=15)
                entry.insert(0, config.default)

            entry.pack(side="left", padx=4)
            self.param_entries[config.name] = entry

        tk.Button(self.param_frame, text="Update Parameters",
                  command=lambda: self.update_parameters(model_name)).pack(pady=5, padx=10)

    def update_model_and_metric_options(self, event=None) -> None:
        """Update model and metric options based on problem type."""
        ptype = self.problem_choice.get()
        models = Config.MODEL_OPTIONS.get(ptype, [])

        self.model_choice['values'] = models
        if models:
            self.model_choice.set(models[0])
            self.update_param_inputs(models[0])

        self.metric_choice['values'] = Config.METRIC_OPTIONS.get(ptype, [])
        if Config.METRIC_OPTIONS.get(ptype):
            self.metric_choice.set(Config.METRIC_OPTIONS[ptype][0])

        is_clustering = ptype == "Clustering"
        self.target_label.config(state="disabled" if is_clustering else "normal")
        self.target_menu.config(state="disabled" if is_clustering else "readonly")
        self.test_size_label.config(state="disabled" if is_clustering else "normal")
        self.test_size_entry.config(state="disabled" if is_clustering else "normal")
        self.scaler_choice.config(state="normal")

    def on_model_change(self, event=None) -> None:
        """Handle model change event."""
        self.update_param_inputs(self.model_choice.get())

    def initialize_options(self) -> None:
        """Initialize GUI options."""
        self.update_model_and_metric_options()

    def run_model(self) -> None:
        """Run the selected model and display results."""
        if not self.dataset.is_loaded():
            messagebox.showerror("Error", "Load a dataset first.")
            return

        try:
            result = self._execute_model()
            self._display_results(result)
        except ValueError as ve:
            logging.error(f"Value error during model execution: {str(ve)}")
            messagebox.showerror("Value Error", str(ve))
        except Exception as e:
            logging.error(f"Unexpected error during model execution: {str(e)}")
            messagebox.showerror("Error", f"Model execution failed:\n{e}")

    def _prepare_data_for_execution(self, ptype: str, target: Optional[str], test_size: float) -> Tuple:
        """Prepare data for model execution"""
        if ptype == "Deep":  # Apply prepare_features only for ANN
            X, scaler = self.dataset.prepare_features(target)
            y_raw = self.dataset.data[target]
            if y_raw.isna().any():
                raise ValueError("Target column contains null values. Handle nulls first.")

            # Encode target for ANN
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)
            class_map = {i: c for i, c in enumerate(le.classes_)}
            y = to_categorical(y, num_classes=len(le.classes_))

            if X.shape[0] == 0 or y.shape[0] == 0:
                raise ValueError("No data available after preprocessing.")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            return X_train, X_test, y_train, y_test, class_map

        # For non-ANN models
        df = self.dataset.data.copy()
        if ptype != "Clustering":
            if not target or target not in df.columns:
                raise ValueError("Select a valid target column.")
            X = df.drop(columns=[target])
            y_raw = df[target]
            if y_raw.isna().any():
                raise ValueError("Target column contains null values. Handle nulls first.")
        else:
            X = df
            y_raw = None

        # Apply LabelEncoder to categorical columns in X for non-ANN models
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        if ptype == "Clustering":
            num_cols = X.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("No numeric columns available for clustering after encoding.")
            if X[num_cols].isna().any().any():
                raise ValueError("Numeric columns contain null values. Handle nulls first.")
            # Apply scaling to numeric columns for Clustering
            if len(num_cols) > 0:
                scaler = StandardScaler() if self.dataset.scaler_type == "StandardScaler" else MinMaxScaler()
                X[num_cols] = scaler.fit_transform(X[num_cols])
            return X, None, None, None, None

        # Handle target encoding for Classification and Regression
        class_map = None
        if ptype == "Regression":
            y = pd.to_numeric(y_raw, errors='coerce')
            if y.isna().any():
                raise ValueError("Target column contains non-numeric values that could not be coerced.")
            valid_idx = y.notna()
            X, y = X.loc[valid_idx], y.loc[valid_idx]
        else:  # Classification
            # Ensure target has more than one unique value
            if y_raw.nunique() <= 1:
                raise ValueError("Target column has only one unique value, unsuitable for classification.")
            # Always apply LabelEncoder to ensure integer values
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)
            class_map = {i: c for i, c in enumerate(le.classes_)}
            y = y.astype(int)  # Ensure integer values

        if X.shape[0] == 0 or (y is not None and y.shape[0] == 0):
            raise ValueError("No data available after preprocessing.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test, class_map

    def _execute_model(self) -> Dict[str, Any]:
        """Execute the selected model based on problem type."""
        ptype = self.problem_choice.get().strip()
        model_name = self.model_choice.get().strip()
        metric = self.metric_choice.get().strip()

        if not all([ptype, model_name, metric]):
            raise ValueError("Choose problem type, model, and metric.")

        params = self.current_params.get(model_name, {})
        target = self.target_menu.get().strip() if ptype != "Clustering" else None
        if ptype != "Clustering":
            if not target or target not in self.dataset.columns:
                raise ValueError("Select a valid target column.")
            test_size = Validator.validate_test_size(self.test_size_entry.get())
        else:
            test_size = 0.0

        if model_name == "ANN":
            X_temp, _ = self.dataset.prepare_features(target)
            params['input_size'] = X_temp.shape[1]  # Flattened feature dimension
            params['num_labels'] = len(np.unique(self.dataset.data[target]))

        model = ModelFactory.create_model(model_name, params)
        args = self._prepare_data_for_execution(ptype, target, test_size)

        if ptype == "Clustering":
            results = self.runners["Clustering"].run(model, args[0], metric=metric)
        elif ptype == "Classification":
            results = self.runners["Classification"].run(model, *args[:4], metric, args[4])
        elif ptype == "Deep":
            results = self.runners["Deep"].run(
                model, *args[:4], metric=metric, class_map=args[4],
                epochs=params.get("epochs", 20), batch_size=params.get("batch_size", 128)
            )
        else:  # Regression
            results = self.runners["Regression"].run(model, *args[:4], metric)

        logging.info(f"Model {model_name} executed for {ptype} with metric {metric}")

        return {
            "problem_type": ptype,
            "model_name": model_name,
            "params": params,
            "results": results,
            "metric": metric
        }

    def _display_results(self, execution_result: Dict[str, Any]) -> None:
        """Display model results"""
        output_lines = [
            f"=== {execution_result['problem_type']} Results ===\n",
            f"Model: {execution_result['model_name']}\n"
        ]

        results = execution_result['results']
        for key, value in results.items():
            output_lines.append(f"{key}: {value}")

        output_lines.append("\nModel executed successfully.")
        messagebox.showinfo("Model Results", "\n".join(output_lines))

        # Visualize confusion matrix
        if execution_result['metric'] == "Confusion Matrix" and "Confusion Matrix" in results:
            cm = np.array(results["Confusion Matrix"])
            labels = results.get("Label Mapping", None)

    def exit_application(self) -> None:
        """Exit the application"""
        result = messagebox.askyesno("Exit", "Are you sure you want to exit?")
        if result:
            self.root.quit()
            self.root.destroy()

    def run(self) -> None:
        """Run the main GUI loop."""
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)
        self.root.mainloop()

# ================= MAIN =================
if __name__ == "__main__":
    app = MLApp()
    app.run()