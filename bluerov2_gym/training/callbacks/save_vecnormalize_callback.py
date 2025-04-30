from stable_baselines3.common.base_class import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from pathlib import Path


class SaveVectorNormalizationCallback(BaseCallback):

    def __init__(self, verbose=0, save_path=None):
        super().__init__(verbose)
        self.save_path = Path(save_path) / "best_vector_norm.pkl"
        self.verbose = verbose

    def _on_step(self) -> bool:

        assert isinstance(
            self.training_env, VecNormalize
        ), "Training environment is not wrapped in VecNormalize"

        self.training_env.save(self.save_path)

        if self.verbose > 0:
            print(f"Saved vector normalization to {self.save_path}")

        return True
