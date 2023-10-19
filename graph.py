from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import plotly as px

class Grapher:
    def __init__ (self, data : pd.DataFrame):
        self.data = data
    
    def corr_matrix(self):
        correlation_matrix = self.corr()
        plt.figure(figsize=(12, 10))
        with sns.axes_style("white"):
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f",
                        annot_kws={"size": 8}, xticklabels=True, yticklabels=True, cbar=False)
            plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    def plot_predictions(model, X_test, y_test, title):
        y_pred = model.predict(X_test)
        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Real Values', 'y': 'Predicted Values'},
                        title=title)
        fig.update_layout(showlegend=False)
        fig.add_shape(type='line', x0=min(y_test), x1=max(y_test), y0=min(y_test), y1=max(y_test))
        fig.show()