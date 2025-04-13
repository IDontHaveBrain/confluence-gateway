import typer

from . import generate_commands, index_commands, search_commands

app = typer.Typer(
    help="Confluence Gateway CLI - Search, Index, and Generate.",
    no_args_is_help=True,
)

app.add_typer(search_commands.app, name="search", help="Search Confluence content.")
app.add_typer(index_commands.app, name="index", help="Manage content indexing.")
app.add_typer(
    generate_commands.app, name="generate", help="Generate answers using RAG."
)

if __name__ == "__main__":
    app()
