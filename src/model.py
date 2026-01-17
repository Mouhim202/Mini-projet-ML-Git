from sklearn.linear_model import LogisticRegression

def build_model(cfg):
    return LogisticRegression(
        max_iter=cfg["model"]["max_iter"]
    )
