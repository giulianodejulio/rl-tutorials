"""Output locations shared by the tutorial experiments and plot scripts."""

from pathlib import Path


def figure_path(topic, filename):
    directory = Path(__file__).resolve().parent / "results" / topic / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / Path(filename).name


def data_path(figure):
    directory = Path(figure).parent.parent / "data"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / Path(figure).with_suffix(".npz").name
