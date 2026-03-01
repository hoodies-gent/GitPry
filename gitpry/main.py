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
    from gitpry.git_utils.repository import get_recent_commits, build_prompt_context
    from gitpry.llm.client import stream_ollama
    from gitpry.llm.prompts import SYSTEM_PROMPT, build_user_prompt
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown

    console = Console()
    
    # 1. Extract Git History
    with console.status("[bold blue]Scanning local Git repository...", spinner="dots"):
        commits = get_recent_commits(limit=limit)
        
    if not commits:
        # Error is already logged by the git utility
        raise typer.Exit(code=1)
        
    context_str = build_prompt_context(commits)
    prompt = build_user_prompt(context_str, question)
    
    # 2. Query LLM
    logger.debug("Prompt constructed. Querying local Ollama model...")
    console.print(f"\n[bold green]GitPry[/] is analyzing your last {len(commits)} commits...\n")
    
    generator = stream_ollama(prompt=prompt, system=SYSTEM_PROMPT)
    
    if not generator:
        # Error is already logged by the llm client
        raise typer.Exit(code=1)
        
    # 3. Stream Response
    response_text = ""
    # We use rich Live to render Markdown dynamically as it streams in!
    with Live(Markdown(response_text), console=console, refresh_per_second=10) as live:
        for chunk in generator:
            response_text += chunk
            live.update(Markdown(response_text))
            
    console.print()  # Add a trailing newline when done

def cli():
    app()

if __name__ == '__main__':
    cli()
