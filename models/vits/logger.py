import wandb
import numpy as np
from sklearn.metrics import classification_report, accuracy_score


class Logger:
    def __init__(self, hps):
        self.results = {}
        self.reset_results()

        wandb.init(
            id=hps.train.run_id,
            project=hps.train.project_name,
            config=hps.dictionary(),
            name=hps.train.run_name,
            mode=hps.train.mode,
            resume=hps.train.resume,
        )

        hps.train.run_id = wandb.run.id
        hps.train.run_name = wandb.run.name

    def __del__(self):
        self.finish()

    def reset_results(self):
        self.results = {
            'train': {},
            'test': {},
        }

    def _process_dict(self, container, data):
        for k, v in data.items():
            if isinstance(v, dict):
                if k not in container:
                    container[k] = {}
                self._process_dict(container[k], v)
            else:
                if k not in container:
                    container[k] = []
                container[k].append(v)

    def _calculate_means(self, data, exclude=[]):
        for k, v in data.items():
            if k in exclude:
                continue
            if isinstance(v, dict):
                self._calculate_means(v)
            else:
                data[k] = np.mean(v)


    def step_train(self, **kwargs):
        self._process_dict(self.results['train'], kwargs)

    def step_test(self, **kwargs):
        self._process_dict(self.results['test'], kwargs)

    def add_audio(self, audio, idx, caption):
        if 'audio' not in self.results['test']:
            self.results['test']['audio'] = {}
        self.results['test']['audio'][idx] = wandb.Audio(audio, 22050, caption)

    def add_image(self, image, idx, caption):
        if 'image' not in self.results['test']:
            self.results['test']['image'] = {}
        self.results['test']['image'][idx] = wandb.Image(image, 'RGB', caption)

    def add_attention(self, attention, idx, caption):
        if 'attention' not in self.results['test']:
            self.results['test']['attention'] = {}
        self.results['test']['attention'][idx] = wandb.Image(attention, 'RGB', caption)

    def log_results(self, epoch, step):
        data = self.results
        for stage, items in data.items():
            self._calculate_means(items, exclude=['audio', 'image', 'attention'])

        wandb.log(data, step=step, commit=False)
        wandb.log({"epoch": epoch})
        self.reset_results()

    def finish(self):
        wandb.finish()
        self.reset_results()
