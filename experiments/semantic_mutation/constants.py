import hashlib
import logging
from pathlib import Path
import sys
from typing import List
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import RobertaTokenizerFast

# This is a hack, to locate xpgaur.
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(project_root)

from xpgaur.models.Li import pre_process_for_li
from xpgaur.utils.constants import ProjectPaths

logger = logging.getLogger(__name__)


class MyAutoEncoder(nn.Module):
    # From: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/linear-autoencoder/Simple_Autoencoder_Solution.ipynb
    def __init__(self, input_dim):
        super(MyAutoEncoder, self).__init__()

        self.input_dim = input_dim
        self._inter_dim_1 = int(0.67 * input_dim)
        self._inter_dim_2 = int(0.33 * input_dim)
        logger.info(
            f"Autoencoder dimensions - input: {input_dim}, "
            f"inter1: {self._inter_dim_1}, inter2: {self._inter_dim_2}."
        )

        # encoder
        self.fc1 = nn.Linear(input_dim, self._inter_dim_1)
        self.fc2 = nn.Linear(self._inter_dim_1, self._inter_dim_2)

        ## decoder ##
        self.fc3 = nn.Linear(self._inter_dim_2, self._inter_dim_1)
        self.fc4 = nn.Linear(self._inter_dim_1, self.input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        encoded = F.relu(self.fc2(x))

        x = F.relu(self.fc3(encoded))
        decoded = F.sigmoid(self.fc4(x))
        return decoded

    def decision_function(
        self, features: np.ndarray, is_tensor: bool = False
    ) -> np.ndarray:
        """Compute anomaly scores using MSE for reconstruction error scores.

        We manually define this function to possess the same behavior as
        sklearn-based model to keep the same training functions.

        Args:
            features (np.ndarray):

        Returns:
            np.ndarray: _description_
        """
        if not is_tensor:
            test_data = torch.tensor(features, dtype=torch.float32)
        else:
            test_data = features

        self.eval()
        with torch.no_grad():
            recon = self(test_data)
            mse_per_sample = F.mse_loss(recon, test_data, reduction="none").mean(
                dim=1
            )
            recon_errors = mse_per_sample.cpu().numpy()
        scores = -recon_errors
        return scores


class MyAutoEncoderRelu(nn.Module):
    # From: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/linear-autoencoder/Simple_Autoencoder_Solution.ipynb
    def __init__(self, input_dim):
        super(MyAutoEncoderRelu, self).__init__()

        self.input_dim = input_dim
        self._inter_dim_1 = int(0.67 * input_dim)
        self._inter_dim_2 = int(0.33 * input_dim)
        logger.info(
            f"Autoencoder dimensions - input: {input_dim}, "
            f"inter1: {self._inter_dim_1}, inter2: {self._inter_dim_2}."
        )

        # encoder
        self.fc1 = nn.Linear(input_dim, self._inter_dim_1)
        self.fc2 = nn.Linear(self._inter_dim_1, self._inter_dim_2)

        ## decoder ##
        self.fc3 = nn.Linear(self._inter_dim_2, self._inter_dim_1)
        self.fc4 = nn.Linear(self._inter_dim_1, self.input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        encoded = F.relu(self.fc2(x))

        x = F.relu(self.fc3(encoded))
        decoded = F.relu(self.fc4(x))
        # decoded = F.sigmoid(self.fc4(x))
        return decoded

    def decision_function(
        self, features: np.ndarray, is_tensor: bool = False
    ) -> np.ndarray:
        """Compute anomaly scores using MSE for reconstruction error scores.

        We manually define this function to possess the same behavior as
        sklearn-based model to keep the same training functions.

        Args:
            features (np.ndarray):

        Returns:
            np.ndarray: _description_
        """
        if not is_tensor:
            test_data = torch.tensor(features, dtype=torch.float32)
        else:
            test_data = features

        self.eval()
        with torch.no_grad():
            recon = self(test_data)
            mse_per_sample = F.mse_loss(recon, test_data, reduction="none").mean(
                dim=1
            )
            recon_errors = mse_per_sample.cpu().numpy()
        scores = -recon_errors
        return scores


class MyAutoEncoderTanh(nn.Module):
    # From: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/linear-autoencoder/Simple_Autoencoder_Solution.ipynb
    def __init__(self, input_dim):
        super(MyAutoEncoderTanh, self).__init__()

        self.input_dim = input_dim
        self._inter_dim_1 = int(0.67 * input_dim)
        self._inter_dim_2 = int(0.33 * input_dim)
        logger.info(
            f"Autoencoder dimensions - input: {input_dim}, "
            f"inter1: {self._inter_dim_1}, inter2: {self._inter_dim_2}."
        )

        # encoder
        self.fc1 = nn.Linear(input_dim, self._inter_dim_1)
        self.fc2 = nn.Linear(self._inter_dim_1, self._inter_dim_2)

        ## decoder ##
        self.fc3 = nn.Linear(self._inter_dim_2, self._inter_dim_1)
        self.fc4 = nn.Linear(self._inter_dim_1, self.input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        encoded = F.relu(self.fc2(x))

        x = F.relu(self.fc3(encoded))
        # decoded = F.relu(self.fc4(x))
        decoded = F.tanh(self.fc4(x))
        return decoded

    def decision_function(
        self, features: np.ndarray, is_tensor: bool = False
    ) -> np.ndarray:
        """Compute anomaly scores using MSE for reconstruction error scores.

        We manually define this function to possess the same behavior as
        sklearn-based model to keep the same training functions.

        Args:
            features (np.ndarray):

        Returns:
            np.ndarray: _description_
        """
        if not is_tensor:
            test_data = torch.tensor(features, dtype=torch.float32)
        else:
            test_data = features

        self.eval()
        with torch.no_grad():
            recon = self(test_data)
            mse_per_sample = F.mse_loss(recon, test_data, reduction="none").mean(
                dim=1
            )
            recon_errors = mse_per_sample.cpu().numpy()
        scores = -recon_errors
        return scores


class BaseSecureBERT:
    """Share logic for all SecureBERT models.

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
        device: torch.device,
        project_paths: ProjectPaths,
        bert_model: str,
        batch_size: int,
    ):
        self.device = device
        # We require a project_path object to find cached embeddings.
        self.project_paths = project_paths
        self.bert_model = bert_model
        self.batch_size = batch_size

        # This way, the same embeddings are generated for the same sentence
        # Indeed, some layers are randomly initialized as the warning states.
        torch.manual_seed(2)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.bert_model)
        self.rb_model = transformers.RobertaModel.from_pretrained(self.bert_model)
        self.rb_model.to(self.device)
        self.rb_model.eval()

        self.model_name = None
        self.clf = None

    # Shared proprocessing functions
    def _cache_path(self, df: pd.DataFrame) -> str:
        hash_val = hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()
        return os.path.join(
            self.project_paths.embeddings_path,
            f"embeddings-{hash_val}.pkl",
        )

    def _load_or_compute_embeddings(
        self, df: pd.DataFrame, batch_size: int
    ) -> List[np.ndarray]:
        # We are iterating over different embeddings each time, it does not make 
        # sense to save them.
        # cache_path = self._cache_path(df)
        # if os.path.isfile(cache_path):
        #     logger.info(f"Loaded cached SBERT embeddings from {cache_path}")
        #     return pd.read_pickle(cache_path)

        queries = df["full_query"].values
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = queries[i : i + batch_size].tolist()
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.rb_model(**inputs, output_hidden_states=True)
                batch_embeddings = outputs.pooler_output.cpu().numpy()
                embeddings.extend(batch_embeddings)


        # pd.to_pickle(embeddings, cache_path)
        return embeddings

    def preprocess(self, df: pd.DataFrame, batch_size: int = 64):
        embeddings = self._load_or_compute_embeddings(df, batch_size)
        labels = df["label"].to_numpy()
        return embeddings, labels


class AutoEncoder_SecureBERT(BaseSecureBERT):
    def __init__(
        self,
        device: torch.device,
        project_paths: ProjectPaths,
        bert_model: str = "ehsanaghaei/SecureBERT",
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 16,
    ):
        super().__init__(device, project_paths, bert_model, batch_size)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def load_model(self, model_name: str):
        self.model_name = model_name

        models_paths = self.project_paths.models_paths

        for save_dir in models_paths:
            model_path = f"{save_dir}{self.model_name}.pth"
            meta_path = f"{save_dir}{self.model_name}_meta.pth"

            if Path(model_path).exists() and Path(meta_path).exists():
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)

                self.learning_rate = metadata.get(
                    "learning_rate", self.learning_rate
                )
                self.epochs = metadata.get("epochs", self.epochs)
                self.batch_size = metadata.get("batch_size", self.batch_size)
                input_dim = metadata["input_dim"]

                self.clf = MyAutoEncoderTanh(input_dim=input_dim)
                self.clf.to(self.device)

                state_dict = torch.load(model_path, map_location=self.device)
                self.clf.load_state_dict(state_dict)

                self.clf.eval()
                logger.info(f"Loaded AutoEncoder model from {model_path}")
                return

        raise FileNotFoundError(
            f"Model '{self.model_name}' not found in any of the search paths:\n"
            + "\n".join(f"  - {path}" for path in models_paths)
        )

    def save_model(self):
        save_dir = self.project_paths.models_path
        model_path = f"{save_dir}{self.model_name}.pth"
        meta_path = f"{save_dir}{self.model_name}_meta.pth"

        torch.save(self.clf.state_dict(), model_path)
        metadata = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "input_dim": next(self.clf.parameters()).shape[1],
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

    # TODO: Rename preprocess_for_preds into preprocess ?
    # Wrapper to fit to preprocessing_generic_ae function call.
    def preprocess_for_preds(self, df: pd.DataFrame):
        return self.preprocess(df=df)

    def X_to_tensor(self, X) -> torch.Tensor:
        """
        Used during testing.

        Args:
            df (_type_): _description_
        Returns:
            _type_: _description_
        """
        X_array = np.array(X)
        X_tensors = torch.FloatTensor(X_array).to(self.device)
        return X_tensors

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
    ):
        self.model_name = model_name
        embeddings, _ = self.preprocess(df=df)

        # Init variables for training + model
        input_dim = len(embeddings[0])
        # Because embeddings have values between -1 and 1, we use an autoencoder with tanh

        self.clf = MyAutoEncoderTanh(
            input_dim=input_dim,
        )
        self.clf.to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.learning_rate)
        X_tensor = self.X_to_tensor(embeddings)

        self.clf.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i : i + self.batch_size]
                batch = batch.to(self.device)

                optimizer.zero_grad()
                reconstructed = self.clf(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.debug(
                f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(X_tensor):.6f}"
            )
        self.save_model()


class AutoEncoder_Li:
    def __init__(
        self,
        device,
        project_paths: ProjectPaths,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        use_scaler: bool = False,
    ):
        self.clf = None
        self.model_name = None
        self.project_paths = project_paths

        # Let's use MaxAbsScaler => 0 and 1 because no value can be negative here.
        self._scaler = MaxAbsScaler()

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_scaler = use_scaler
        self.device = device

        self.feature_columns = None

    def preprocess_for_preds(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        df_pped = df.copy()
        labels = np.array(df_pped["label"])
        df_pped = pre_process_for_li(df_pped)
        return df_pped, labels

    def X_to_tensor(self, X) -> torch.Tensor:
        """
        Used during testing.

        Args:
            df (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 4096.

        Returns:
            _type_: _description_
        """
        X = X.values
        if self.use_scaler:
            X = self._scaler.transform(X)
        X_tensors = torch.FloatTensor(X).to(self.device)
        return X_tensors

    def preprocess_for_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for training. We ignore label data.

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df_pped, _ = self.preprocess_for_preds(df=df)
        return df_pped

    def load_model(self, model_name: str):
        self.model_name = model_name

        models_paths = self.project_paths.models_paths

        for save_dir in models_paths:
            model_path = f"{save_dir}{self.model_name}.pth"
            meta_path = f"{save_dir}{self.model_name}_meta.pth"

            if Path(model_path).exists() and Path(meta_path).exists():
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)

                self.learning_rate = metadata.get("learning_rate")
                self.epochs = metadata.get("epochs")
                self.batch_size = metadata.get("batch_size")
                self.use_scaler = metadata.get("use_scaler")
                self.feature_columns = metadata.get("feature_columns")

                input_dim = len(self.feature_columns)

                if self.use_scaler:
                    self._scaler = metadata.get("scaler")
                    self.clf = MyAutoEncoder(
                        input_dim=input_dim,
                    )
                else:
                    self.clf = MyAutoEncoderRelu(input_dim=input_dim)
                self.clf.to(self.device)
                self.clf.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.clf.eval()
                return

        raise FileNotFoundError(
            f"Model '{self.model_name}' not found in any of the search paths:\n"
            + "\n".join(f"  - {path}" for path in models_paths)
        )

    def save_model(self):
        save_dir = self.project_paths.models_path
        model_path = f"{save_dir}{self.model_name}.pth"
        meta_path = f"{save_dir}{self.model_name}_meta.pth"

        torch.save(self.clf.state_dict(), model_path)
        metadata = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "use_scaler": self.use_scaler,
            "feature_columns": self.feature_columns,
            "scaler": self._scaler if self.use_scaler else None,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved model to: {model_path} and metadata to: {meta_path}")

    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        # Init variables for training + model
        self.model_name = model_name
        df_pped = self.preprocess_for_train(df)
        self.feature_columns = df_pped.columns.tolist()
        input_dim = len(self.feature_columns)

        # Let's apply Scaler here and not in preprocess, as we want to keep
        # information about the columns
        df_pped = np.array(df_pped)
        assert df_pped.min() >= 0

        # If scaling =>
        if self.use_scaler:
            scaled_data = self._scaler.fit_transform(df_pped)
            self._scaler_min = scaled_data.min(axis=None)
            self._scaler_max = scaled_data.max(axis=None)
            train_data = torch.FloatTensor(scaled_data)
            self.clf = MyAutoEncoder(
                input_dim=input_dim,
            )
        else:
            train_data = torch.FloatTensor(df_pped)
            self.clf = MyAutoEncoderRelu(input_dim=input_dim)

        self.clf.to(self.device)

        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.learning_rate)

        self.clf.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i : i + self.batch_size]
                batch = batch.to(self.device)

                optimizer.zero_grad()
                reconstructed = self.clf(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.debug(
                f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(train_data):.6f}"
            )
        self.save_model()
