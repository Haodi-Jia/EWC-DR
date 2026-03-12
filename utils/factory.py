def get_model(model_name, args):
    name = model_name.lower()
    if name == "ewc":
        from models.ewc import EWC
        return EWC(args)
    elif name == "finetune":
        from models.finetune import Finetune
        return Finetune(args)
    elif name == "ewc_dr":
        from models.ewc_dr import EWCDR
        return EWCDR(args)
    elif name == "ewc_mas":
        from models.ewc_mas import EWC_MAS
        return EWC_MAS(args)
    elif name == "ewc_online":
        from models.ewc_online import OnlineEWC
        return OnlineEWC(args)
    elif name == "ewc_si":
        from models.ewc_si import EWC_SI
        return EWC_SI(args)
    else:
        assert 0
