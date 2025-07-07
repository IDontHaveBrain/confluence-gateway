#!/usr/bin/env python3

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from atlassian import Confluence
from confluence_formatter import ConfluenceFormatter
from rich.console import Console
from rich.progress import track
from rich.prompt import Confirm

console = Console()
app = typer.Typer()


class RealDataGenerator:
    def __init__(self, config_path: Path = Path("config")):
        self.config_path = config_path

        with open(config_path / "production_config.yaml") as f:
            self.production_config = yaml.safe_load(f)

        self.index_path = config_path / "real_data" / "content_index.json"
        if self.index_path.exists():
            with open(self.index_path) as f:
                self.index = json.load(f)
        else:
            self.index = {"entries": []}

        self.confluence = Confluence(
            url=os.environ.get("CONFLUENCE_URL"),
            username=os.environ.get("CONFLUENCE_USERNAME"),
            password=os.environ.get("CONFLUENCE_API_TOKEN"),
            cloud=True,
        )

        self.tracking_file = config_path / "confluence_generation_tracking.json"
        self.created_spaces = []
        self.created_pages = []
        self.formatter = ConfluenceFormatter()
        self.load_tracking()

    def load_tracking(self):
        if self.tracking_file.exists():
            with open(self.tracking_file) as f:
                tracking = json.load(f)
                self.created_spaces = tracking.get("spaces", [])
                self.created_pages = tracking.get("pages", [])

    def save_tracking(self):
        tracking = {
            "spaces": self.created_spaces,
            "pages": self.created_pages,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.tracking_file, "w") as f:
            json.dump(tracking, f, indent=2)

    def get_content_by_category(
        self, category: str, limit: int
    ) -> list[dict[str, Any]]:
        entries = []
        for entry in self.index["entries"]:
            if category in entry["categorization"]["categories"]:
                entries.append(entry)
        random.shuffle(entries)
        return entries[:limit]

    def create_space(
        self, space_config: dict[str, Any], dry_run: bool = False
    ) -> str | None:
        prefix = self.production_config["generation"]["prefix"]
        space_key = f"{prefix}{space_config['key']}"
        space_name = f"{prefix} - {space_config['name']}"

        if dry_run:
            console.print(
                f"[yellow]DRY RUN: Would create space {space_key} - {space_name}[/yellow]"
            )
            return space_key

        try:
            existing = self.confluence.get_space(space_key, expand="description.plain")
            if existing:
                console.print(f"[yellow]Space {space_key} already exists[/yellow]")
                return space_key
        except:
            pass
        console.print(f"[blue]Creating space {space_key} - {space_name}[/blue]")
        try:
            space = self.confluence.create_space(space_key, space_name)

            space_id = None
            if space:
                if isinstance(space, dict):
                    space_id = space.get("id")
                else:
                    space_id = getattr(space, "id", None)

            if not space_id:
                try:
                    created_space = self.confluence.get_space(space_key)
                    space_id = created_space.get("id") if created_space else space_key
                except:
                    space_id = space_key

            self.created_spaces.append(
                {
                    "key": space_key,
                    "id": space_id,
                    "created_at": datetime.now().isoformat(),
                }
            )

            return space_key

        except Exception as e:
            console.print(f"[red]Error creating space {space_key}: {e}[/red]")
            return None

    def format_content_for_confluence(self, entry: dict[str, Any]) -> str:
        # Pass the base directory for markdown files
        markdown_base_dir = self.config_path / "real_data"
        return self.formatter.format_entry_for_confluence(entry, markdown_base_dir)

    def create_pages_in_space(
        self, space_key: str, space_config: dict[str, Any], dry_run: bool = False
    ):
        pages_count = space_config["pages_count"]
        if os.environ.get("TEST_MODE") == "1":
            pages_count = min(3, pages_count)
        all_entries = []
        for category in space_config["categories"]:
            entries = self.get_content_by_category(
                category, pages_count // len(space_config["categories"]) + 5
            )
            all_entries.extend(entries)
        all_entries = all_entries[:pages_count]

        console.print(
            f"\n[cyan]Creating {len(all_entries)} pages in space {space_key}[/cyan]"
        )

        for entry in track(all_entries, description=f"Creating pages in {space_key}"):
            title = (
                f"{self.production_config['generation']['prefix']} - {entry['title']}"
            )

            if dry_run:
                console.print(f"[yellow]DRY RUN: Would create page '{title}'[/yellow]")
                continue

            try:
                existing_pages = self.confluence.get_all_pages_from_space(
                    space_key,
                    start=0,
                    limit=1,
                    status="current",
                    expand="",
                    content_type="page",
                )

                page_exists = False
                for existing_page in existing_pages:
                    if existing_page.get("title") == title:
                        page_exists = True
                        break

                if page_exists:
                    console.print(f"[yellow]Page '{title}' already exists[/yellow]")
                    continue

                content = self.format_content_for_confluence(entry)

                page = self.confluence.create_page(
                    space=space_key, title=title, body=content
                )

                self.created_pages.append(
                    {
                        "id": page["id"],
                        "space": space_key,
                        "title": title,
                        "source_entry_id": entry["id"],
                        "created_at": datetime.now().isoformat(),
                    }
                )

                time.sleep(0.5)

            except Exception as e:
                console.print(f"[red]Error creating page '{title}': {e}[/red]")

    def generate_all(self, dry_run: bool = False, recreate: bool = False):
        if recreate and not dry_run:
            if Confirm.ask("[red]This will delete existing test data. Continue?[/red]"):
                self.cleanup()
            else:
                return

        spaces = self.production_config["generation"]["spaces"]
        console.print(
            "[bold blue]Creating Confluence Test Data from Real Documentation[/bold blue]\n"
        )

        if os.environ.get("TEST_MODE") == "1":
            spaces = spaces[:1]
            console.print("[yellow]TEST MODE: Creating only first space[/yellow]\n")

        for space_config in spaces:
            space_key = self.create_space(space_config, dry_run)
            if space_key:
                self.create_pages_in_space(space_key, space_config, dry_run)

        if not dry_run:
            self.save_tracking()

        self.display_summary()

    def cleanup(self, pattern: str | None = None):
        if not pattern:
            pattern = self.production_config["generation"]["prefix"]

        console.print(f"\n[red]Cleaning up data with pattern '{pattern}'...[/red]")

        if not self.created_spaces and not self.created_pages:
            console.print(
                f"[yellow]No tracked data found. Searching for spaces with pattern '{pattern}'...[/yellow]"
            )
            self.cleanup_by_pattern(pattern)
            return
        deleted_pages = 0
        for page_info in self.created_pages[:]:
            try:
                self.confluence.remove_page(page_info["id"])
                console.print(f"[green]✓ Deleted page: {page_info['title']}[/green]")
                self.created_pages.remove(page_info)
                deleted_pages += 1
            except Exception as e:
                console.print(
                    f"[yellow]Could not delete page {page_info['title']}: {e}[/yellow]"
                )
        deleted_spaces = 0
        for space_info in self.created_spaces[:]:
            try:
                space_key = space_info["key"]
                console.print(
                    f"[yellow]Attempting to delete space {space_key}...[/yellow]"
                )

                try:
                    response = self.confluence.delete(f"rest/api/space/{space_key}")
                    console.print(
                        f"[green]✓ Space '{space_key}' deleted successfully![/green]"
                    )
                    self.created_spaces.remove(space_info)
                    deleted_spaces += 1
                except Exception as delete_error:
                    try:
                        console.print(
                            f"[yellow]Delete failed, attempting to archive space {space_key}...[/yellow]"
                        )
                        self.confluence.archive_space(space_key)
                        console.print(f"[green]✓ Space '{space_key}' archived[/green]")
                        self.created_spaces.remove(space_info)
                        deleted_spaces += 1
                    except Exception:
                        console.print(
                            f"[yellow]Could not delete or archive space {space_key}: {delete_error}[/yellow]"
                        )
                        console.print(
                            "[yellow]This may be due to API limitations or permissions.[/yellow]"
                        )
                        self.created_spaces.remove(space_info)

            except Exception as e:
                console.print(f"[red]Error with space {space_info['key']}: {e}[/red]")

        self.save_tracking()

        console.print(f"\n[green]Deleted {deleted_pages} pages[/green]")
        console.print(f"[green]Deleted/archived {deleted_spaces} spaces[/green]")

    def cleanup_by_pattern(self, pattern: str):
        try:
            all_spaces = []
            start = 0
            limit = 100

            while True:
                try:
                    response = self.confluence.get_all_spaces(start=start, limit=limit)

                    if not response or "results" not in response:
                        break

                    spaces = response.get("results", [])

                    for space in spaces:
                        space_key = space.get("key", "")
                        if pattern in space_key:
                            all_spaces.append(space)
                            console.print(
                                f"  • Found: {space_key} - {space.get('name', 'Unknown')}"
                            )

                    if "next" not in response.get("_links", {}) or len(spaces) < limit:
                        break

                    start += limit

                except Exception as e:
                    console.print(f"[red]Error fetching spaces: {e}[/red]")
                    break

            if not all_spaces:
                console.print(
                    f"[yellow]No spaces found matching pattern '{pattern}'[/yellow]"
                )
                return

            console.print(
                f"\n[bold]Found {len(all_spaces)} spaces matching pattern '{pattern}'[/bold]"
            )
            deleted_spaces = 0
            total_deleted_pages = 0

            for space in all_spaces:
                space_key = space["key"]
                space_name = space.get("name", "Unknown")

                try:
                    console.print(
                        f"\n[bold]Processing space: {space_key} - {space_name}[/bold]"
                    )
                    all_pages = []
                    start = 0
                    limit = 50

                    while True:
                        try:
                            pages_response = self.confluence.get_all_pages_from_space(
                                space_key, start=start, limit=limit
                            )

                            if not pages_response:
                                break

                            if isinstance(pages_response, list):
                                pages = pages_response
                            elif (
                                hasattr(pages_response, "get")
                                and "results" in pages_response
                            ):
                                pages = pages_response.get("results", [])
                            else:
                                pages = []

                            if not pages:
                                break

                            all_pages.extend(pages)

                            if len(pages) < limit:
                                break
                            start += limit

                        except Exception as e:
                            console.print(
                                f"[red]Error fetching pages from {space_key}: {e}[/red]"
                            )
                            break
                    deleted_pages = 0
                    for page in all_pages:
                        try:
                            page_id = (
                                page.get("id")
                                if isinstance(page, dict)
                                else getattr(page, "id", None)
                            )
                            if page_id:
                                self.confluence.remove_page(page_id)
                                deleted_pages += 1
                                time.sleep(0.1)
                        except Exception as e:
                            error_msg = str(e).lower()
                            if "not found" not in error_msg and "404" not in error_msg:
                                console.print(
                                    f"[red]Failed to delete page {page_id}: {e}[/red]"
                                )

                    total_deleted_pages += deleted_pages
                    console.print(
                        f"[green]✓ Deleted {deleted_pages} pages from space {space_key}[/green]"
                    )
                    try:
                        response = self.confluence.delete(f"rest/api/space/{space_key}")
                        console.print(
                            f"[green]✓ Space '{space_key}' deleted successfully![/green]"
                        )
                        deleted_spaces += 1
                    except Exception as e:
                        try:
                            self.confluence.archive_space(space_key)
                            console.print(
                                f"[green]✓ Space '{space_key}' archived[/green]"
                            )
                            deleted_spaces += 1
                        except Exception:
                            console.print(
                                f"[yellow]Could not delete or archive space {space_key}: {e}[/yellow]"
                            )
                            if deleted_pages > 0:
                                deleted_spaces += 1

                except Exception as e:
                    console.print(f"[red]Error cleaning space {space_key}: {e}[/red]")

            console.print(
                "\n[bold green]✓ Pattern-based cleanup completed![/bold green]"
            )
            console.print(f"  • Spaces cleaned: {deleted_spaces}/{len(all_spaces)}")
            console.print(f"  • Pages deleted: {total_deleted_pages}")

        except Exception as e:
            console.print(f"[red]Error during pattern cleanup: {e}[/red]")

    def display_summary(self):
        console.print("\n[bold]Generation Summary[/bold]")
        console.print(f"Spaces created/used: {len(self.created_spaces)}")
        console.print(f"Pages created: {len(self.created_pages)}")
        pages_by_space = {}
        for page in self.created_pages:
            space = page["space"]
            pages_by_space[space] = pages_by_space.get(space, 0) + 1

        console.print("\n[bold]Pages per space:[/bold]")
        for space, count in pages_by_space.items():
            console.print(f"  {space}: {count} pages")


@app.command()
def create(
    dry_run: bool = typer.Option(False, "--dry-run"),
    recreate: bool = typer.Option(False, "--recreate"),
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
):
    generator = RealDataGenerator(config_dir)
    generator.generate_all(dry_run, recreate)


@app.command()
def cleanup(
    pattern: str = typer.Option(None, "--pattern", "-p"),
    confirm: bool = typer.Option(False, "--confirm"),
    config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c"),
):
    generator = RealDataGenerator(config_dir)

    if not confirm:
        if not Confirm.ask("[red]This will delete test data. Continue?[/red]"):
            return

    generator.cleanup(pattern)


@app.command()
def status(config_dir: Path = typer.Option(Path("config"), "--config-dir", "-c")):
    generator = RealDataGenerator(config_dir)

    console.print("[bold]Generated Content Status[/bold]\n")

    if not generator.created_spaces and not generator.created_pages:
        console.print("No content has been generated yet.")
        return

    console.print(f"Total spaces: {len(generator.created_spaces)}")
    console.print(f"Total pages: {len(generator.created_pages)}")

    if generator.created_spaces:
        console.print("\n[bold]Spaces:[/bold]")
        for space in generator.created_spaces[:5]:
            console.print(f"  - {space['key']} (created: {space['created_at']})")
        if len(generator.created_spaces) > 5:
            console.print(f"  ... and {len(generator.created_spaces) - 5} more")

    pages_by_space = {}
    for page in generator.created_pages:
        space = page["space"]
        pages_by_space[space] = pages_by_space.get(space, 0) + 1

    if pages_by_space:
        console.print("\n[bold]Pages by space:[/bold]")
        for space, count in pages_by_space.items():
            console.print(f"  - {space}: {count} pages")


if __name__ == "__main__":
    app()
