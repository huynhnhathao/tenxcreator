# core/app/app.py
from flask import Flask, send_from_directory
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

# Initialize Flask server and Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Serve static audio files from the artifact directory
@server.route("/audio/<path:filename>")
def serve_audio(filename):
    """Serve audio files from the artifact directory."""
    return send_from_directory("./artifact/audio", filename)


# Define app layout with URL-based routing
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)


# Callback to render page content based on URL
@callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    """Render the appropriate page based on the URL pathname."""
    if pathname == "/audio-generation":
        from core.app.pages.audio_generation import layout

        return layout
    else:
        return html.H1("404 - Page Not Found")


if __name__ == "__main__":
    app.run_server(debug=True)
