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

from gitpry.config import settings

@app.command()
def ask(
    question: str = typer.Argument(..., help="The natural language question to ask about your git history."),
    limit: int = typer.Option(settings.git.limit, "--limit", help="Limit the number of commits to analyze.")
):
    """
    Ask questions about your git history.
    """
    from gitpry.git_utils.repository import get_recent_commits, build_prompt_context, count_tokens
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
        
    # Calculate base tokens consumed by the skeleton prompt itself
    base_skeleton = build_user_prompt("", question) + SYSTEM_PROMPT
    base_tokens = count_tokens(base_skeleton)
        
    context_str, included_count = build_prompt_context(
        commits, 
        max_tokens=settings.llm.max_tokens, 
        base_tokens=base_tokens
    )
    
    # 2. Safety Check
    if included_count == 0:
        from rich.panel import Panel
        err_msg = (
            f"The current `max_tokens` limit (**{settings.llm.max_tokens}**) is too small "
            f"to fit even a single recent commit alongside its diff patch.\n\n"
            f"**How to fix this:**\n"
            f"1. Increase `max_tokens` in your `.gitpry.toml` (e.g., to 6000 or 8000), OR\n"
            f"2. Disable diff extraction by setting `include_diff = false`."
        )
        console.print(Panel(Markdown(err_msg), title="[bold red]Token Limit Overflow", border_style="red"))
        raise typer.Exit(code=1)
        
    prompt = build_user_prompt(context_str, question)
    
    # 3. Query LLM
    logger.debug(f"Prompt constructed ({count_tokens(prompt)} tokens). Querying local Ollama model...")
    
    msg = f"\n[bold green]GitPry[/] is analyzing your last {included_count} commits"
    if included_count < len(commits):
        msg += f" [yellow](truncated {len(commits) - included_count} to fit {settings.llm.max_tokens} token limit)[/]"
    console.print(msg + "...\n")
    
    generator = stream_ollama(prompt=prompt, system=SYSTEM_PROMPT)
    
    if not generator:
        # Error is already logged by the llm client
        raise typer.Exit(code=1)
        
    # 3. Stream Response with Cold Start Polish
    response_text = ""
    first_chunk_received = False
    
    # Start a persistent spinner to mask the LLM 'Wake Up' and Attention latency (TTFT)
    with console.status("[bold magenta]Awakening LLM and processing context...", spinner="bouncingBar"):
        for chunk in generator:
            if not first_chunk_received:
                first_chunk_received = True
                # The LLM has finally responded! Break out of the spinner context 
                # so we can transition to the Markdown streaming view.
                response_text += chunk
                break
                
    if first_chunk_received:
        # We use rich Live to render Markdown dynamically as the REST of the chunks stream in
        with Live(Markdown(response_text), console=console, refresh_per_second=10) as live:
            for chunk in generator:
                response_text += chunk
                live.update(Markdown(response_text))
                
        console.print()  # Add a trailing newline when done
    else:
        # Failsafe if generator yielded nothing
        console.print("[red]The LLM returned an empty response.[/red]")

def cli():
    app()

if __name__ == '__main__':
    cli()
