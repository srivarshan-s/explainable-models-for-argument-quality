from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def display_metrics(pred, target):
    print("METRICS\tSCORE")

    r2_val = r2_score(target, pred)
    print("R2:", end="\t")
    print(f"{r2_val:>.4f}")

    mae_val = mean_absolute_error(target, pred)
    print("MAE:", end="\t")
    print(f"{mae_val:>.4f}")

    mse_val = mean_squared_error(target, pred, squared=True)
    print("MSE:", end="\t")
    print(f"{mse_val:>.4f}")

    rmse_val = mean_squared_error(target, pred, squared=False)
    print("RMSE:", end="\t")
    print(f"{rmse_val:>.4f}")

    pearson_corr, _ = pearsonr(target, pred)
    print("Pcorr:", end="\t")
    print(f"{pearson_corr:>.4f}")

    spearman_corr, _ = spearmanr(target, pred)
    print("Scorr:", end="\t")
    print(f"{spearman_corr:>.4f}")
