from multiprocessing.pool import ThreadPool


from tqdm import tqdm
import torch
import argbind
from frechet_audio_distance import FrechetAudioDistance, load_audio_task
from audiotools.core.util import find_audio


@argbind.bind(without_prefix=True)
@torch.no_grad()
def evaluate_fad(
    input: str = "samples/input",
    output: str = "samples/output",
    n_proc: int = 50,
):
    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
        verbose=True,
        audio_load_worker=n_proc,
    )

    fad_score = frechet.score(input, output, dtype="float32")

    print(f"FAD score: {fad_score}")


if __name__ == "__main__":
    args = argbind.parse_args()
    print(args)
    with argbind.scope(args):
        evaluate_fad()
