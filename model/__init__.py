from .nos import NosTokenizer, Nos, NosPredictor

model_dict = {
    'nos_tokenizer': NosTokenizer,
    'nos': Nos,
    'nos_predictor': NosPredictor
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError


