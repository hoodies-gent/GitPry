import typer
from gitpry.utils.logger import logger, setup_logger

app = typer.Typer(help="GitPry: Talk to your Git history. Reclaim the Who, When, and Why behind every line of code.", no_args_is_help=True)

def version_callback(value: bool):
    if value:
        import importlib.metadata
        version = importlib.metadata.version("git-pry")
        typer.echo(f"GitPry CLI Version: {version}")
        raise typer.Exit()

@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose debug logging."),
    version: bool = typer.Option(None, "--version", callback=version_callback, is_eager=True, help="Show the version and exit."),
):
    if verbose:
        setup_logger(debug=True)
        logger.debug("Verbose logging enabled.")

@app.command()
def ask(
    question: str = typer.Argument(..., help="The natural language question to ask about your git history."),
    limit: int = typer.Option(500, "--limit", help="Limit the number of commits to analyze.")
):
    """
    Ask questions about your git history.
    """
    # This is a placeholder
    logger.info("The 'ask' command is not fully implemented yet.")
    typer.echo(f"Hello from GitPry! You asked: '{question}' with limit {limit}.")

def cli():
    app()

if __name__ == '__main__':
    cli()
