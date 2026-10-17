"""Folder loader and original category-specific audio prompts for EmoS."""

import json
import os
import re
from pathlib import Path

RUBRICS = json.loads(Path(__file__).with_name("rubrics.json").read_text(encoding="utf-8"))
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
RESPONSE_PATTERN = re.compile(r"^Model_(\d+)(?:$|[_\s.-])", re.IGNORECASE)
SPEECH_PATTERN = re.compile(r"^(user|model)_speech(\d+)(?:$|[_\s.-])", re.IGNORECASE)


def category_for_path(scene, root):
    parts = (root.name,) + scene.relative_to(root).parts
    for part in reversed(parts):
        normalized = re.sub(r"[^a-z0-9]", "", part.lower())
        matches = [
            code for code, config in RUBRICS.items()
            if any(normalized.startswith(re.sub(r"[^a-z0-9]", "", prefix.lower()))
                   for prefix in config["prefixes"])
        ]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous category: {scene}")
        if matches:
            return matches[0]
    return None


def load_tasks(data_root, categories):
    """Read scored standard-test folders; reject incomplete or ambiguous scenes."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Data root is not a directory: {root}")
    selected = set(categories)
    if not selected or selected - RUBRICS.keys():
        raise ValueError(f"Choose categories from: {', '.join(RUBRICS)}")
    tasks = []

    def walk_error(error):
        raise error

    for directory, subdirs, filenames in os.walk(root, onerror=walk_error):
        subdirs[:] = sorted(name for name in subdirs if not name.startswith("."))
        scene = Path(directory)
        audio_files = [scene / name for name in sorted(filenames)
                       if Path(name).suffix.lower() in AUDIO_EXTENSIONS]
        if not audio_files:
            continue
        code = category_for_path(scene, root)
        if code is None:
            if any(RESPONSE_PATTERN.match(path.stem) for path in audio_files):
                raise ValueError(f"Scored audio folder has an unknown category: {scene}")
            continue
        if code not in selected:
            continue
        responses = {}
        for path in audio_files:
            match = RESPONSE_PATTERN.match(path.stem)
            if match:
                score = int(match.group(1))
                if score not in range(1, 5) or score in responses:
                    raise ValueError(f"Invalid or duplicate Model score: {path}")
                responses[score] = path
        if set(responses) != {1, 2, 3, 4}:
            raise ValueError(f"Expected exactly Model_1 through Model_4 in {scene}")

        def relative(path):
            resolved = path.resolve()
            if not resolved.is_file() or not resolved.is_relative_to(root):
                raise ValueError(f"Audio must be a file inside data root: {path}")
            return path.relative_to(root).as_posix()

        context = {}
        if RUBRICS[code]["multi_turn"]:
            turns = []
            for path in audio_files:
                match = SPEECH_PATTERN.match(path.stem)
                if match:
                    turns.append((int(match.group(2)), match.group(1).lower(), path))
            turns.sort(key=lambda turn: turn[0])
            if not turns or turns[-1][1] != "user":
                raise ValueError(f"Multi-turn scene must end with user_speechN: {scene}")
            if len({turn[0] for turn in turns}) != len(turns):
                raise ValueError(f"Speech sequence numbers must be unique: {scene}")
            context["history_context_audios"] = [
                {"role": role, "path": relative(path)} for _, role, path in turns[:-1]
            ]
            context["final_user_audio"] = relative(turns[-1][2])
        else:
            users = [path for path in audio_files if path.name.lower().startswith("user_")]
            if len(users) != 1:
                raise ValueError(f"Expected exactly one user audio in {scene}")
            context["user_audio"] = relative(users[0])
        scene_id = scene.relative_to(root).as_posix()
        for score, response in sorted(responses.items()):
            tasks.append({
                "id": f"{code}/{scene_id}/model_{score}",
                "category": code,
                "sub_dir": scene_id,
                **context,
                "model_response_audio": relative(response),
                "true_score": score,
            })
    missing = selected - {task["category"] for task in tasks}
    if missing:
        raise ValueError(f"No standard-test samples found for: {', '.join(sorted(missing))}. "
                         "Use extracted standard-test audio, not blinded human-evaluation data.")
    return tasks


def build_message(task, data_root, think=False, think_max_len=50):
    """Keep rubric and message wording from the original single/multi-turn scripts."""
    config = RUBRICS[task["category"]]
    base_question = "Based on the rubric provided above, please evaluate the quality of the 'Model Response' and assign a score (1-4)."
    if think:
        limit = f" (less than {think_max_len} words)" if think_max_len > 0 else ""
        question = f"{base_question} Please output your thinking and analysis process in <think> </think>{limit}, and then output ONLY the final numerical score (1, 2, 3, or 4) in <answer> </answer>."
    else:
        question = f"{base_question} Please output ONLY the final numerical score in <answer> </answer>."
    content = [{"type": "text", "text": f"You are an expert audio dialogue evaluation assistant. Here is the scoring rubric:\n{config['rubric']}\n\n"}]

    def text(value):
        content.append({"type": "text", "text": value})

    def audio(value):
        content.append({"type": "audio", "audio": str(Path(data_root).expanduser().resolve() / value)})

    if config["multi_turn"]:
        text("--- Dialogue History ---\n")
        for item in task["history_context_audios"]:
            text("User: " if item["role"] == "user" else "Model: ")
            audio(item["path"])
            text("\n")
        text("\n--- Dialogue to Evaluate ---\n")
        text("Latest User Speech: ")
        audio(task["final_user_audio"])
        text("\nModel Response to Evaluate: ")
    else:
        text("Latest User Speech:")
        audio(task["user_audio"])
        text("\nModel Response to Evaluate:")
    audio(task["model_response_audio"])
    text(f"\n\n{question}")
    return [{"role": "user", "content": content}]
