from src.train import train_model

def test_training_log_reg():
    model = train_model("log_reg")
    assert model is not None

def test_training_rf():
    model = train_model("rf")
    assert model is not None
