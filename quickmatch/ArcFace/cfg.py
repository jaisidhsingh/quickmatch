from types import SimpleNamespace

cfg = SimpleNamespace(**{})


cfg.device = "cuda"
cfg.weights = "./backbone.pth"
cfg.network = "r50"


cfg.output_path = "./test.pickle"
# setting a threshold to give us somewhat 
# symmetric results from deepface
# other implementations of arcface vary in 
# thresholds. might have to adjust
cfg.threshold = 0.76