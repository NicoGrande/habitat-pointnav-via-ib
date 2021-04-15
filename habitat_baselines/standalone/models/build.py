from .odometer_new import Net

def build_egomotion_estimation_model(cfg, device):
    model = Net(cfg)
    model = model.to(device)
    return model