# core/app/pages/audio_generation.py
import os
import base64
import threading
import datetime
from pydantic import ValidationError
import dash
from dash import html, dcc, callback, Input, Output, State, MATCH, ALL
from core.api import text_to_audio
from core.api.models import TextToAudioInput, RawAudioPrompt

# Register this page with Dash
dash.register_page(__name__, path="/audio-generation")

# Directory paths
AUDIO_SAVE_PATH = "./artifact/audio"
PROMPT_DIR = "./prompts"
TEMP_PROMPT_DIR = "./artifact/temp_prompts"

# Ensure directories exist
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)
os.makedirs(TEMP_PROMPT_DIR, exist_ok=True)

# Load pre-existing prompt options from PROMPT_DIR
prompt_options = [{"label": "None", "value": "none"}]
if os.path.exists(PROMPT_DIR):
    for file in os.listdir(PROMPT_DIR):
        if file.endswith(".wav"):
            prompt_name = file[:-4]
            prompt_options.append({"label": prompt_name, "value": prompt_name})

# Define page layout
layout = html.Div(
    [
        html.H1("Text to Audio Generation"),
        html.Div(
            [
                # Left: Input controls
                html.Div(
                    [
                        dcc.Input(
                            id="text-input",
                            type="text",
                            placeholder="Enter text to generate audio",
                            style={
                                "width": "100%",
                                "padding": "10px",
                                "margin-bottom": "10px",
                            },
                        ),
                        html.Div(
                            [
                                html.P("Select a prompt:"),
                                dcc.Dropdown(
                                    id="prompt-selection",
                                    options=prompt_options,
                                    value="none",
                                    style={"width": "100%"},
                                ),
                            ],
                            style={"margin-bottom": "10px"},
                        ),
                        html.Div(
                            [
                                html.P("Or upload a prompt audio:"),
                                dcc.Upload(
                                    id="prompt-upload",
                                    children=html.Button(
                                        "Upload Prompt Audio",
                                        style={"margin-bottom": "5px"},
                                    ),
                                    multiple=False,
                                ),
                                dcc.Input(
                                    id="prompt-transcript",
                                    type="text",
                                    placeholder="Enter transcript of prompt audio",
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "margin-top": "5px",
                                    },
                                ),
                            ],
                            style={"margin-bottom": "10px"},
                        ),
                        html.Button(
                            "Generate Audio",
                            id="generate-button",
                            style={"width": "100%", "padding": "10px"},
                        ),
                        html.Div(id="generation-status", style={"margin-top": "10px"}),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "padding": "20px",
                        "vertical-align": "top",
                    },
                ),
                # Right: Generated audio list
                html.Div(
                    [
                        html.H2("Generated Audio Files"),
                        dcc.Interval(
                            id="audio-list-interval", interval=1000, n_intervals=0
                        ),
                        html.Div(
                            id="audio-list",
                            style={
                                "height": "80vh",
                                "overflow-y": "scroll",
                                "padding": "10px",
                            },
                        ),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "padding": "20px",
                        "vertical-align": "top",
                    },
                ),
            ],
            style={"display": "flex"},
        ),
    ]
)


# Background thread for audio generation
def generate_audio_thread(input_data: TextToAudioInput):
    """
    Generate and save audio in a background thread.

    Args:
        input_data (TextToAudioInput): Validated input data for audio generation.
    """
    try:
        audio_arrays = text_to_audio(input_data)
        for i, audio_array in enumerate(audio_arrays):
            # Placeholder: Save audio array as WAV (replace with real audio saving logic)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(input_data.save_path, f"audio_{timestamp}_{i}.wav")
            with open(save_path, "w") as f:
                f.write("Simulated audio content")  # Replace with actual WAV writing
    except Exception as e:
        print(f"Error in audio generation thread: {e}")


# Callback to start audio generation
@callback(
    Output("generation-status", "children"),
    Input("generate-button", "n_clicks"),
    State("text-input", "value"),
    State("prompt-upload", "contents"),
    State("prompt-upload", "filename"),
    State("prompt-transcript", "value"),
    State("prompt-selection", "value"),
)
def start_generation(
    n_clicks,
    text,
    prompt_contents,
    prompt_filename,
    prompt_transcript,
    prompt_selection,
):
    """Start audio generation based on user inputs."""
    if n_clicks is None:
        return ""
    if not text:
        return "Please enter some text."

    try:
        # Handle audio prompt
        audio_prompt = None
        if prompt_contents:
            # Decode and save uploaded prompt audio
            content_type, content_string = prompt_contents.split(",")
            decoded = base64.b64decode(content_string)
            temp_prompt_path = os.path.join(TEMP_PROMPT_DIR, prompt_filename)
            with open(temp_prompt_path, "wb") as f:
                f.write(decoded)
            audio_prompt = RawAudioPrompt(
                transcript=prompt_transcript,
                audio_file_path=temp_prompt_path,
                sample_rate=24000,
                channels=1,
                max_duration=10,
            )
        elif prompt_selection != "none":
            # Use pre-existing prompt
            prompt_audio_path = os.path.join(PROMPT_DIR, f"{prompt_selection}.wav")
            transcript_path = os.path.join(PROMPT_DIR, f"{prompt_selection}.txt")
            with open(transcript_path, "r") as f:
                transcript = f.read().strip()
            audio_prompt = RawAudioPrompt(
                transcript=transcript,
                audio_file_path=prompt_audio_path,
                sample_rate=24000,
                channels=1,
                max_duration=10,
            )

        # Create validated input data
        input_data = TextToAudioInput(
            texts=[text], audio_prompt=audio_prompt, save_path=AUDIO_SAVE_PATH
        )

        # Start generation in a background thread
        thread = threading.Thread(target=generate_audio_thread, args=(input_data,))
        thread.start()
        return "Generation started."
    except ValidationError as e:
        return f"Validation error: {e}"
    except Exception as e:
        return f"Error: {e}"


# Function to list generated audio files
def get_audio_list():
    """Generate a list of audio file components for display."""
    audio_files = sorted(
        [f for f in os.listdir(AUDIO_SAVE_PATH) if f.endswith(".wav")], reverse=True
    )
    audio_components = []
    for audio_file in audio_files:
        audio_path = f"/audio/{audio_file}"
        audio_components.append(
            html.Div(
                [
                    html.Audio(src=audio_path, controls=True, style={"width": "100%"}),
                    dcc.Input(
                        id={"type": "rename-input", "index": audio_file},
                        value=audio_file,
                        style={"width": "70%", "margin": "5px"},
                    ),
                    html.Button(
                        "Rename",
                        id={"type": "rename-button", "index": audio_file},
                        style={"margin": "5px"},
                    ),
                    html.Button(
                        "Delete",
                        id={"type": "delete-button", "index": audio_file},
                        style={"margin": "5px"},
                    ),
                    html.P(
                        f"Created: {audio_file.split('_')[1]}", style={"margin": "5px"}
                    ),
                ],
                style={
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "margin-bottom": "10px",
                },
            )
        )
    return audio_components


# Callback to update audio list
@callback(Output("audio-list", "children"), Input("audio-list-interval", "n_intervals"))
def update_audio_list(n):
    """Update the displayed list of generated audio files."""
    return get_audio_list()


# Callback to handle audio deletion
@callback(
    Output("audio-list", "children"),
    Input({"type": "delete-button", "index": ALL}, "n_clicks"),
    State({"type": "delete-button", "index": ALL}, "id"),
)
def delete_audio(n_clicks, ids):
    """Delete an audio file when the delete button is clicked."""
    for n, id_dict in zip(n_clicks, ids):
        if n:
            audio_file = id_dict["index"]
            os.remove(os.path.join(AUDIO_SAVE_PATH, audio_file))
    return get_audio_list()


# Callback to handle audio renaming
@callback(
    Output("audio-list", "children"),
    Input({"type": "rename-button", "index": ALL}, "n_clicks"),
    State({"type": "rename-input", "index": ALL}, "value"),
    State({"type": "rename-button", "index": ALL}, "id"),
)
def rename_audio(n_clicks, new_names, ids):
    """Rename an audio file when the rename button is clicked."""
    for n, new_name, id_dict in zip(n_clicks, new_names, ids):
        if n:
            old_file = id_dict["index"]
            new_file = new_name if new_name.endswith(".wav") else new_name + ".wav"
            os.rename(
                os.path.join(AUDIO_SAVE_PATH, old_file),
                os.path.join(AUDIO_SAVE_PATH, new_file),
            )
    return get_audio_list()
