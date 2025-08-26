import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
import cv2
import numpy as np
from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import PowerTransformer, KBinsDiscretizer


class DataframeAnalyzer:
    def __init__(self, df : pd.DataFrame):
        self._df = df
        self._numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        self._non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()

    def drop_features(self, features: list):
        self._df.drop(columns=features, inplace=True)
        self._numeric_columns = self._df.select_dtypes(include=['number']).columns.tolist()
        self._non_numeric_columns = self._df.select_dtypes(exclude=['number']).columns.tolist()

    def get_missing_value_count(self, column_name: str) -> int:
        return self._df[column_name].isnull().sum()
    
    def get_column_type(self, column_name: str) -> str:
        return self._df[column_name].dtype

    def get_true_count(self, column_name: str) -> int:
        return self._df[column_name].count()
    
    def get_unique_value_count(self, column_name: str) -> int:
        return self._df[column_name].nunique()
    
    def get_value_counts(self, column_name: str) -> pd.Series:
        return self._df[column_name].value_counts()
    
    def get_non_numeric_columns(self) -> list:
        return self._non_numeric_columns
    
    def get_numeric_columns(self) -> list:
        return self._numeric_columns
    
    def get_skewed(self):
        skew = self._df.select_dtypes(include=['number']).skew()
        return skew

    def log_transform_skewed_values(self, skewed_cols: list, threshold: float = 0.5):
        for col in skewed_cols:
            if abs(self.get_skewed()[col]) >= threshold:
                self._df[col] = np.log1p(self._df[col])

    def power_transform(self, cols: list, method: str = 'yeo-johnson'):
        for col in cols:
            if col in self._numeric_columns:
                scaler = PowerTransformer(method=method)
                self._df[col] = scaler.fit_transform(self._df[[col]])


    def otsu_thresholding(self, cols: list):
        kb = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy="kmeans")
        for col in cols:
            self._df[col] = kb.fit_transform(self._df[[col]])

    def visualize_and_return_missing_values_per_column(self, type: str = "pie", size : tuple = (10, 5), missing_threshold: int = 0) -> list:
        missing_value_cols = []
        for col in self._df.columns:
            missing_count = self.get_missing_value_count(col)
            true_count = self.get_true_count(col)
            total = missing_count + true_count
            if missing_count == 0:
                continue
            if missing_count / total > missing_threshold:
                missing_value_cols.append(col)
            match type:
                case "pie":
                    plt.figure(figsize=size)
                    plt.pie(
                        [missing_count, true_count],
                        labels=["Missing", "Present"],
                        autopct="%1.1f%%",
                        colors=["red", "blue"],
                        startangle=90
                    )
                    plt.title("Missing Values in Column: " + col)
                    plt.tight_layout()
                    plt.show()
        return missing_value_cols

    def visualize_numeric_columns(self, ylabel : str = None, size: tuple = (10, 5), type : str = "box", custom_columns : list = [], unique_value_threshhold : int = 5):
        if custom_columns:
            for col in custom_columns:
                unique_values = self.get_unique_value_count(col)
                if unique_values > unique_value_threshhold:
                    self._graph_maker(type=type, col=col, ylabel=ylabel, size=size)
        else:
            for col in self._numeric_columns:
                unique_values = self.get_unique_value_count(col)
                if unique_values > unique_value_threshhold:
                    self._graph_maker(type=type, col=col, ylabel=ylabel, size=size)

    def visualize_column_value_counts(self, size: tuple = (10, 5), unique_value_threshold: int = 20):
        for col in self._df.columns:
            unique_values = self.get_unique_value_count(col)
            if unique_values < unique_value_threshold:
                counts = self.get_value_counts(col)
                plt.figure(figsize=size)
                counts.plot(kind="bar", color="skyblue")
                plt.title(f"Value Count of {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.xticks(rotation=45)
                plt.show()

    def separate_based_on_label(self, y_label:str):
        return self._df.groupby(y_label)

    def visualize_correlation_numeric(self, size: tuple = (10, 5)):
        corr = self._df[self._numeric_columns].corr()
        plt.figure(figsize=size)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Matrix")
        plt.show()

    def visualize_correlation_categorical(self, ylabel: str, size: tuple = (10, 5)):
        for col in self._non_numeric_columns:
            contingency = pd.crosstab(self._df[ylabel], self._df[col])
            self._graph_maker(type="heatmap", size=size, col=col, data=contingency)

    def anova_test(self, ylabel : str, col : str):
        if col== ylabel:
            return None
        df_byGenre = self.separate_based_on_label(ylabel)
        groups = [df[col] for genre, df in df_byGenre]
        stat, p = f_oneway(*groups)
        return stat, p
    
    def mutual_info(self, ylabel : str):
        X = self._df.select_dtypes(include=['number'])
        y = self._df[ylabel]

        mi = mutual_info_classif(X,y, discrete_features='auto')
        for col, score in zip(X, mi):
            print(f"mutal info {col}: {score:.3f}")

    def get_df(self):
        return self._df

    def _graph_maker(self, type: str, size: tuple, col: str = None, ylabel: str = None, data : pd.DataFrame = None):
        match type:
            case "box":
                plt.figure(figsize=size)
                plt.title(f"Box Plot of {col}")
                sns.boxplot(data=self._df, x=ylabel, y=col)

            case "histo":
                plt.figure(figsize=size)
                sns.histplot(data=self._df, x=col, hue=ylabel, multiple="stack", kde=True, common_norm=False)
                plt.title(f"Histogram of {col}")
                plt.xlabel(col)
                plt.ylabel("Count")

            case "heatmap":
                plt.figure(figsize=size)
                sns.heatmap(data, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Genre vs {col}")

            case "pair":
                plt.figure(figsize=size)
                sns.pairplot(self._df, hue=ylabel, vars=self._numeric_columns, plot_kws={"alpha": 0.3, "s": 5, "marker" : 'o'})
                plt.xlabel(ylabel)
                plt.ylabel("Features")

        plt.tight_layout()
        plt.show()
    
