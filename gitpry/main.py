import typer
from typing import Optional
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
    limit: int = typer.Option(settings.git.limit, "--limit", help="Limit the number of commits to analyze (legacy fallback)."),
    top_k: int = typer.Option(5, "--top-k", help="Number of semantically similar commits to retrieve from the RAG index."),
    no_rag: bool = typer.Option(False, "--no-rag", help="Bypass the RAG index and use legacy chronological retrieval."),
    branch: Optional[str] = typer.Option(None, "--branch", help="Filter RAG results to commits indexed from this branch. Default: search all branches."),
):
    """
    Ask questions about your git history.
    
    If a local RAG index exists (built with `git pry index`), this will perform
    fast semantic retrieval to find the most relevant commits for your question.
    Otherwise, it falls back to chronological commit scraping.
    """
    from gitpry.llm.client import stream_ollama
    from gitpry.llm.prompts import SYSTEM_PROMPT, build_user_prompt
    from gitpry.rag.vector_store import get_repo_id, get_db_path, TABLE_NAME, search_similar
    from gitpry.git_utils.repository import get_repo_stats, format_repo_stats_block
    from gitpry.rag.query_router import classify_query
    from gitpry.git_utils.scanner import scan_structured
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    import lancedb

    console = Console()

    # ── Always collect repo stats first (ground truth for aggregate queries) ─
    with console.status("[dim]Gathering repository stats...[/dim]", spinner="dots"):
        stats = get_repo_stats(".", branch=branch or "HEAD")
        repo_stats_block = format_repo_stats_block(stats)

    # ── Step 1: Classify query intent ──────────────────────────────────────
    # TODO(V0.5): Skip classify_query when --no-rag is set; the result is unused in that path.
    with console.status("[dim]Classifying query intent...[/dim]", spinner="dots"):
        route = classify_query(
            question,
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            timeout=settings.llm.timeout,
        )
    logger.debug(f"Query route: {route}")

    # ── Step 2b: Determine RAG vs Legacy ────────────────────────────────────
    repo_id = get_repo_id(".")
    db_path = get_db_path(repo_id)
    db = lancedb.connect(str(db_path))
    has_index = not no_rag and settings.rag.enabled and (TABLE_NAME in db.table_names())

    context_str = ""
    run_legacy = False

    if route == "conversational":
        console.print("\n[dim]Conversational query — bypassing commit retrieval.[/dim]\n")
    elif route == "structured" and not no_rag:
        from gitpry.git_utils.scanner import scan_structured
        console.print("\n[bold green]GitPry[/] (Structured mode) — Scanning commits with filters...\n")
        with console.status("[bold blue]Scanning local Git repository...", spinner="dots"):
            context_str, filter_desc = scan_structured(question, repo_path=".", branch=branch or "HEAD")
        console.print(f"[dim]Applied filters: {filter_desc}[/dim]\n")
    elif has_index:
        # ── RAG PATH ──────────────────────────────────────────────────────
        from gitpry.rag.embedder import get_embedding

        console.print(f"\n[bold green]GitPry[/] (RAG mode) — Searching semantic index for relevant commits...\n")

        with console.status("[bold blue]Vectorizing query...", spinner="dots"):
            query_vector = get_embedding(question)

        if not query_vector:
            console.print("[red]Failed to generate query embedding. Falling back to legacy mode.[/red]\n")
            run_legacy = True  # Drop to fallback below
        else:
            with console.status("[bold blue]Searching vector store...", spinner="dots"):
                results = search_similar(".", query_vector, top_k=top_k, branch_filter=branch)

            if not results:
                if branch:
                    console.print(f"[yellow]No matching commits found in the index for branch '{branch}'.")
                    console.print(f"[dim]Run [bold]git pry index --branch {branch}[/bold] to index that branch first.[/dim]")
                else:
                    console.print("[yellow]No matching commits found in the index. Try running `git pry index` first.[/yellow]")
                raise typer.Exit(code=1)

            # Build a context string from the Top-K semantic results
            context_blocks = []
            for r in results:
                block = (
                    f"[{r['commit_hash_short']}] {r['author']} @ {r['date']}\n"
                    f"Message: {r['message']}\n"
                    f"Relevant Context:\n{r['chunk_text']}"
                )
                context_blocks.append(block)

            context_str = "\n\n---\n\n".join(context_blocks)
            branch_label = f" (branch: {branch})" if branch else " (all branches)"
            console.print(f"[dim]Retrieved {len(results)} semantically relevant chunks{branch_label}.[/dim]\n")
    else:
        run_legacy = True

    if run_legacy:
        # ── LEGACY FALLBACK PATH ──────────────────────────────────────────
        from gitpry.git_utils.repository import get_recent_commits, build_prompt_context, count_tokens

        branch_display = branch or "HEAD"
        console.print(f"\n[bold yellow]GitPry[/] (Legacy mode) — Analyzing last {limit} commits from [cyan]{branch_display}[/cyan].\n[dim]💡 Tip: Run [bold]git pry index[/bold] once to enable fast semantic search across full history.[/dim]\n")

        with console.status("[bold blue]Scanning local Git repository...", spinner="dots"):
            commits = get_recent_commits(limit=limit, branch=branch_display)

        if not commits:
            raise typer.Exit(code=1)

        base_skeleton = build_user_prompt("", question) + SYSTEM_PROMPT
        base_tokens = count_tokens(base_skeleton)

        context_str, included_count = build_prompt_context(
            commits,
            max_tokens=settings.llm.max_tokens,
            base_tokens=base_tokens
        )

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

        msg = f"[bold green]GitPry[/] is analyzing your last {included_count} commits"
        if included_count < len(commits):
            msg += f" [yellow](truncated {len(commits) - included_count} to fit {settings.llm.max_tokens} token limit)[/]"
        console.print(msg + "...\n")

    # ── Build and stream the prompt ────────────────────────────────────────
    prompt = build_user_prompt(context_str, question, repo_stats_block=repo_stats_block)
    logger.debug(f"Sending prompt to LLM...")


    generator = stream_ollama(prompt=prompt, system=SYSTEM_PROMPT)

    if not generator:
        raise typer.Exit(code=1)

    # Stream response with spinner until first token arrives (TTFT UX)
    response_text = ""
    first_chunk_received = False

    with console.status("[bold magenta]Awakening LLM and processing context...", spinner="bouncingBar"):
        for chunk in generator:
            if not first_chunk_received:
                first_chunk_received = True
                response_text += chunk
                break

    if first_chunk_received:
        with Live(Markdown(response_text), console=console, refresh_per_second=10) as live:
            for chunk in generator:
                response_text += chunk
                live.update(Markdown(response_text))
        console.print()
    else:
        console.print("[red]The LLM returned an empty response.[/red]")


@app.command()
def index(
    limit: int = typer.Option(2000, "--limit", help="Max number of commits to index (use 0 for full history)."),
    include_diffs: bool = typer.Option(True, "--include-diffs/--no-diffs", help="Whether to chunk and embed full diff patches."),
    branch: Optional[str] = typer.Option(None, "--branch", help="Branch to index. Defaults to the current HEAD branch."),
    reindex: bool = typer.Option(False, "--reindex", help="Drop and rebuild the entire index from scratch (required after schema upgrades)."),
):
    """
    Build (or update) the local semantic index for the current repository.
    
    Embeds commit history into a local LanceDB vector store so that
    `git pry ask` can perform fast semantic retrieval instead of dumb chronological scraping.
    """
    import lancedb
    from gitpry.git_utils.repository import get_recent_commits, get_branch_names
    from gitpry.rag.chunker import commits_to_chunks
    from gitpry.rag.embedder import get_embedding
    from gitpry.rag.vector_store import (
        get_repo_id, get_db_path, open_or_create_table, get_indexed_hashes,
        upsert_chunks, check_schema_migration, drop_table
    )
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    console = Console()

    # ── Resolve target branch ─────────────────────────────────────────────
    # TODO(V0.4 - Branch Validation): Validate that `branch` exists in the repo before
    # running the full pipeline. Currently a typo in --branch will hit a cryptic git error.
    import git as _git
    try:
        _repo = _git.Repo(".")
        active_branch = _repo.active_branch.name
    except Exception:
        active_branch = "HEAD"
    target_branch = branch or active_branch

    console.print(f"\n[bold blue]GitPry Indexer[/] — Building local semantic index for branch [cyan]{target_branch}[/cyan]...\n")

    # Step 1: Fetch raw commits (temporarily enable diffs if requested)
    original_include_diff = settings.git.include_diff
    settings.git.include_diff = include_diffs

    fetch_limit = limit if limit > 0 else 10000
    with console.status("[bold blue]Scanning commit history...", spinner="dots"):
        commits = get_recent_commits(limit=fetch_limit, branch=target_branch)
        # Backfill full_hash from GitPython repo for stable chunk IDs
        try:
            import git
            repo = git.Repo(".")
            hash_map = {c.hexsha[:8]: c.hexsha for c in repo.iter_commits(target_branch, max_count=fetch_limit)}
            for c in (commits or []):
                c["full_hash"] = hash_map.get(c["hash"], c["hash"])
        except Exception:
            for c in (commits or []):
                c["full_hash"] = c["hash"]

    settings.git.include_diff = original_include_diff  # Restore original setting

    if not commits:
        console.print("[red]No commits found. Aborting.[/red]")
        raise typer.Exit(code=1)

    # Step 2: Connect to LanceDB; detect outdated schema or --reindex
    repo_id = get_repo_id(".")
    db_path = get_db_path(repo_id)
    db = lancedb.connect(str(db_path))

    if reindex:
        # TODO(V0.4 - Selective Reindex): --reindex currently drops the ENTIRE table,
        # losing index data for all other branches. Future: support --reindex --branch foo
        # to drop and rebuild only one branch's chunks without affecting others.
        console.print("[yellow]--reindex: dropping existing index and rebuilding from scratch...[/yellow]")
        drop_table(db)
    elif check_schema_migration(db):
        console.print(
            "[bold red]⚠ Index schema is outdated (missing 'branch' column from V0.3).[/bold red]\n"
            "[yellow]Run: [bold]git pry index --reindex[/bold] to rebuild the index.[/yellow]"
        )
        raise typer.Exit(code=1)
    
    # We probe the first commit's embedding to learn the vector dimension
    console.print("[dim]Probing embedding model dimension...[/dim]")
    probe_embedding = get_embedding("probe")
    if not probe_embedding:
        console.print("[red]Failed to connect to Ollama or `nomic-embed-text` model is not available.[/red]")
        console.print("[yellow]Please run: [bold]ollama pull nomic-embed-text[/bold][/yellow]")
        raise typer.Exit(code=1)

    vector_dim = len(probe_embedding)
    table = open_or_create_table(db, vector_dim)
    already_indexed = get_indexed_hashes(table)  # Global dedup — branch-agnostic

    # Step 3: Filter to only new commits for this branch
    new_commits = [c for c in commits if c["full_hash"] not in already_indexed]
    if not new_commits:
        console.print(f"[green]✓ Index is already up to date for branch [cyan]{target_branch}[/cyan].[/green] ({len(already_indexed)} commits indexed)")
        raise typer.Exit()

    console.print(f"[green]Found {len(new_commits)} new commits to index[/green] ({len(already_indexed)} already indexed for [cyan]{target_branch}[/cyan]).\n")

    # Step 4: Chunk the new commits, tagging with target branch
    chunks = commits_to_chunks(new_commits, branch=target_branch)
    console.print(f"[dim]Generated {len(chunks)} chunks from {len(new_commits)} commits.[/dim]\n")

    # Step 5: Embed each chunk and collect for batch insert
    embedded_chunks = []
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Embedding {len(chunks)} chunks...", total=len(chunks))

        for chunk in chunks:
            vector = get_embedding(chunk["chunk_text"])
            if vector:
                embedded_chunks.append({**chunk, "vector": vector})
            else:
                failed += 1
            progress.advance(task)

    # Step 6: Persist to LanceDB
    upsert_chunks(table, embedded_chunks)

    console.print(f"\n[bold green]✓ Indexed {len(embedded_chunks)} chunks successfully![/bold green]")
    if failed > 0:
        console.print(f"[yellow]⚠ {failed} chunks failed to embed and were skipped.[/yellow]")
    console.print(f"[dim]Vector store location: {db_path}[/dim]\n")


@app.command()
def chat(
    branch: Optional[str] = typer.Option(None, "--branch", help="Restrict chat to commits indexed from this branch."),
    no_rag: bool = typer.Option(False, "--no-rag", help="Bypass the RAG index and use legacy commits only."),
    max_turns: int = typer.Option(10, "--max-turns", help="Max conversation turns to keep in memory."),
):
    """
    Start an interactive multi-turn chat session about your git history.

    GitPry maintains conversation context across turns, enabling follow-up
    questions like "tell me more about that commit" or "who made those changes?".
    RAG retrieval uses the combined context of recent questions for better results.

    Slash commands: /exit  /clear  /stats  /branch <name>  /mode  /help
    """
    from gitpry.llm.client import stream_ollama_chat
    from gitpry.llm.chat_session import ChatSession
    from gitpry.llm.prompts import SYSTEM_PROMPT
    from gitpry.rag.vector_store import get_repo_id, get_db_path, TABLE_NAME, search_similar
    from gitpry.rag.embedder import get_embedding
    from gitpry.rag.query_router import classify_query
    from gitpry.git_utils.scanner import scan_structured
    from gitpry.git_utils.repository import get_repo_stats, format_repo_stats_block
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.rule import Rule
    import lancedb

    console = Console()

    # ── Session init ──────────────────────────────────────────────────────
    session = ChatSession(branch=branch, no_rag=no_rag, max_turns=max_turns)

    console.print()
    console.print(Rule("[bold cyan]GitPry Chat[/bold cyan]", style="cyan"))
    console.print("Multi-turn git history conversation. Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.")
    console.print(f"Current model: {settings.llm.model}\n")

    # ── Check index availability ──────────────────────────────────────────
    repo_id = get_repo_id(".")
    db_path = get_db_path(repo_id)
    db = lancedb.connect(str(db_path))
    has_index = not no_rag and settings.rag.enabled and (TABLE_NAME in db.table_names())

    if not has_index and not no_rag:
        console.print("[yellow]No local index found — running in Legacy mode.[/yellow]")
        console.print("[dim]Run [bold]git pry index[/bold] for faster semantic search.[/dim]\n")

    # ── REPL loop ─────────────────────────────────────────────────────────
    while True:
        try:
            branch_tag = f"[dim]({session.branch})[/dim] " if session.branch else ""
            raw = Prompt.ask(f"\n{branch_tag}[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not raw:
            continue

        # ── Slash commands ────────────────────────────────────────────────
        if raw.startswith("/"):
            cmd_parts = raw[1:].split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd == "exit":
                console.print("[dim]Session ended.[/dim]")
                break
            elif cmd == "clear":
                session.clear()
                console.print("[green]✓ Conversation history cleared.[/green]")
            elif cmd == "stats":
                with console.status("[dim]Fetching stats...[/dim]", spinner="dots"):
                    stats = get_repo_stats(".", branch=session.branch or "HEAD")
                    block = format_repo_stats_block(stats)
                console.print(Markdown(f"```\n{block}\n```"))
            elif cmd == "branch":
                if not arg:
                    console.print(f"[yellow]Current branch filter: {session.branch or '(all)'}[/yellow]")
                    console.print("[dim]Use [bold]/branch <name>[/bold] to set, [bold]/branch -[/bold] to clear.[/dim]")
                elif arg == "-":
                    session.branch = None
                    session.clear()
                    console.print("[green]✓ Branch filter cleared — searching all branches. History cleared.[/green]")
                else:
                    session.branch = arg
                    session.clear()
                    console.print(f"[green]✓ Switched to branch [cyan]{arg}[/cyan]. History cleared.[/green]")
            elif cmd == "mode":
                mode_str = "Legacy (--no-rag)" if no_rag else ("RAG + Structured" if has_index else "Legacy (no index)")
                console.print(f"[dim]Current mode: [bold]{mode_str}[/bold][/dim]")
            elif cmd == "help":
                console.print(Markdown(
                    "| Command | Description |\n"
                    "|---|---|\n"
                    "| `/exit` | End the session |\n"
                    "| `/clear` | Clear conversation history |\n"
                    "| `/stats` | Show repository statistics |\n"
                    "| `/branch <name>` | Filter to a specific branch |\n"
                    "| `/branch -` | Clear branch filter (search all) |\n"
                    "| `/branch` | Show current branch filter |\n"
                    "| `/mode` | Show current retrieval mode |\n"
                    "| `/help` | Show this help |"
                ))
            else:
                console.print(f"[red]Unknown command: /{cmd}[/red]  Type /help for options.")
            continue

        question = raw

        # ── Gather repo stats on first turn only ──────────────────────────
        if not session.repo_stats_block:
            with console.status("[dim]Gathering repository stats...[/dim]", spinner="dots"):
                stats = get_repo_stats(".", branch=session.branch or "HEAD")
                session.repo_stats_block = format_repo_stats_block(stats)

        # ── Route + Retrieve ──────────────────────────────────────────────
        combined_query = session.build_combined_query(question)

        with console.status("[dim]Classifying query...[/dim]", spinner="dots"):
            route = classify_query(combined_query, base_url=settings.llm.base_url,
                                   model=settings.llm.model, timeout=settings.llm.timeout)
        logger.debug(f"Chat turn route: {route}")

        context_str = ""
        if route == "conversational":
            console.print("[dim]Conversational query — bypassing commit retrieval.[/dim]")

        elif route == "structured" and not no_rag:
            with console.status("[dim]Scanning commits with filters...[/dim]", spinner="dots"):
                context_str, filter_desc = scan_structured(
                    question, repo_path=".", branch=session.branch or "HEAD"
                )
            console.print(f"[dim]Structured scan — {filter_desc}[/dim]")

        elif has_index:
            with console.status("[dim]Searching semantic index...[/dim]", spinner="dots"):
                query_vector = get_embedding(combined_query)
                if query_vector:
                    results = search_similar(".", query_vector,
                                             top_k=settings.rag.top_k,
                                             branch_filter=session.branch)
                    if results:
                        context_blocks = [
                            f"[{r['commit_hash_short']}] {r['author']} @ {r['date']}\n"
                            f"Message: {r['message']}\nContext:\n{r['chunk_text']}"
                            for r in results
                        ]
                        context_str = "\n\n---\n\n".join(context_blocks)
                        branch_info = f" (branch: {session.branch})" if session.branch else " (all branches)"
                        console.print(f"[dim]Retrieved {len(results)} chunks{branch_info}.[/dim]")

        else:
            # Legacy: simple recent-commit scrape
            from gitpry.git_utils.repository import get_recent_commits, build_prompt_context, count_tokens
            from gitpry.llm.prompts import build_user_prompt
            with console.status("[dim]Reading recent commits...[/dim]", spinner="dots"):
                commits = get_recent_commits(limit=settings.git.limit, branch=session.branch or "HEAD")
            if commits:
                base_tokens = count_tokens(build_user_prompt("", question) + SYSTEM_PROMPT)
                context_str, _ = build_prompt_context(commits, settings.llm.max_tokens - base_tokens)

        # ── Build messages and stream response ────────────────────────────
        messages = session.build_messages(context_str, question)

        console.print()
        generator = stream_ollama_chat(messages)
        if not generator:
            console.print("[red]Failed to connect to Ollama.[/red]")
            continue

        response_text = ""
        first_received = False

        with console.status("[bold magenta]Thinking...[/bold magenta]", spinner="bouncingBar"):
            for chunk in generator:
                if not first_received:
                    first_received = True
                    response_text += chunk
                    break

        if first_received:
            console.print("[bold green]GitPry:[/bold green]", highlight=False)
            with Live(Markdown(response_text), console=console, refresh_per_second=10) as live:
                for chunk in generator:
                    response_text += chunk
                    live.update(Markdown(response_text))
            console.print()
        else:
            console.print("[red]Empty response from LLM.[/red]")
            continue

        # ── Store turn in history ─────────────────────────────────────────
        session.add_turn(question, context_str, response_text)


@app.command()
def serve():
    """
    Start the GitPry MCP (Model Context Protocol) Server.
    Provides Git context retrieval tools for advanced AI agents like Cursor or Claude Desktop.
    Connect via stdio transport.
    """
    try:
        from gitpry.mcp_server import serve_stdio
        serve_stdio()
    except ImportError as e:
        console.print(f"[red]Error starting MCP server: {e}[/red]")
        console.print("[dim]Did you install the 'mcp' dependency? Run: `pip install mcp`[/dim]")
        raise typer.Exit(code=1)


def cli():
    app()

if __name__ == '__main__':
    cli()
