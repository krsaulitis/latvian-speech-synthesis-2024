class Config:
    def __init__(self, **kwargs):

        self.run_id = kwargs.get('run_id', None)
        self.run_name = kwargs.get('run_name', None)
        self.project_name = kwargs.get('project_name', 'emotion-bert-en')
        self.model_name = kwargs.get('model_name', 'emotion-bert-en')
        self.description = kwargs.get('description', '')
        self.data_dir = kwargs.get('data_dir', 'data')
        self.data_lang = kwargs.get('data_lang', 'en')
        self.base_model = kwargs.get('base_model', 'bert-base-cased')
        self.base_model_config = kwargs.get('base_model_config', None)
        self.base_model_vocab = kwargs.get('base_model_vocab', 'bert-base-cased')
        self.optimizer = kwargs.get('optimizer', 'Adam')
        self.weight_decay = kwargs.get('weight_decay', 0.0)
        self.warmup = kwargs.get('warmup', None)
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.max_len = kwargs.get('max_len', 64)
        self.learning_rate = kwargs.get('learning_rate', 1e-5)
        self.classifier_dropout = kwargs.get('classifier_dropout', 0.0)
        self.seed = kwargs.get('seed', 42)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
