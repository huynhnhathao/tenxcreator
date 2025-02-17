from dataclasses import dataclass


@dataclass
class ModelPath:
    repo_id: str
    file_name: str


class BarkRemoteModelPaths:
    TEXT_SMALL = ModelPath(repo_id="suno/bark", file_name="text.pt")
    COARSE_SMALL = ModelPath(repo_id="suno/bark", file_name="coarse.pt")
    FINE_SMALL = ModelPath(repo_id="suno/bark", file_name="fine.pt")
    TEXT = ModelPath(repo_id="suno/bark", file_name="text_2.pt")
    COARSE = ModelPath(repo_id="suno/bark", file_name="coarse_2.pt")
    FINE = ModelPath(repo_id="suno/bark", file_name="fine_2.pt")
