from importlib.metadata import version

import typer

from . import (
    generate_commands,
    index_commands,
    model_commands,
    search_commands,
    spaces_commands,
)


def version_callback(value: bool) -> None:
    if value:
        try:
            app_version = version("confluence-gateway")
            typer.echo(f"confluence-gateway {app_version}")
        except Exception:
            typer.echo("confluence-gateway 0.1.0")
        raise typer.Exit()


app = typer.Typer(
    help="Confluence Gateway CLI - Search, Index, and Generate.",
    no_args_is_help=True,
)


@app.callback()
def main(
    version_flag: bool = typer.Option(
        None, "--version", callback=version_callback, is_eager=True, help="Show version"
    ),
) -> None:
    """Confluence Gateway CLI - Search, Index, and Generate."""
    pass


app.add_typer(search_commands.app, name="search", help="Search Confluence content.")
app.add_typer(index_commands.app, name="index", help="Manage content indexing.")
app.add_typer(
    generate_commands.app, name="generate", help="Generate answers using RAG."
)
app.add_typer(spaces_commands.app, name="spaces", help="Manage Confluence spaces.")
app.add_typer(
    model_commands.app,
    name="models",
    help="Check embedding model configuration and compatibility.",
)

if __name__ == "__main__":
    app()
