# app.py
from flask import Flask, send_from_directory
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    use_pages=True,
    pages_folder="core/app/pages",
    suppress_callback_exceptions=True,
)
import core.app.pages.audio_generation


# Serve audio files (if needed)
@server.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory("./artifact/audio", filename)


# Define app layout with page container
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), dash.page_container],
)

if __name__ == "__main__":
    app.run_server(debug=True)
