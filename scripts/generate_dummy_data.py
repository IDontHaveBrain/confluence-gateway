#!/usr/bin/env python3
"""
Confluence Dummy Data Generator

This script generates test data in Confluence for testing search capabilities
(keyword, semantic, CQL, hybrid) of Confluence Gateway.
"""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import typer
import yaml
from pydantic import BaseModel
from rich.console import Console
from rich.progress import track

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.core.config import ConfluenceConfig, confluence_config

console = Console()
app = typer.Typer(help="Generate dummy data in Confluence for testing purposes")

logger = logging.getLogger(__name__)

class DummyDataConfig(BaseModel):
    """Configuration for dummy data generation"""
    
    prefix: str = "TESTDUM"
    spaces: List[Dict[str, Any]] = [
        {
            "key": "TECH",
            "name": "Technical Documentation Test",
            "pages_count": 50,
            "categories": ["installation", "architecture", "troubleshooting"]
        },
        {
            "key": "API",
            "name": "API Documentation Test",
            "pages_count": 30,
            "categories": ["rest", "graphql", "webhooks"]
        },
        {
            "key": "KB",
            "name": "Knowledge Base Test",
            "pages_count": 40,
            "categories": ["how-to", "faq", "best-practices"]
        },
        {
            "key": "PROJECT",
            "name": "Project Documentation Test",
            "pages_count": 30,
            "categories": ["planning", "meeting-notes", "releases"]
        },
        {
            "key": "MULTILANG",
            "name": "Multilingual Test",
            "pages_count": 20,
            "categories": ["english", "korean", "mixed"]
        }
    ]
    
    content: Dict[str, Any] = {
        "languages": ["en", "ko", "mixed"],
        "min_words": 200,
        "max_words": 2000,
        "include_code_blocks": True,
        "include_tables": True,
        "include_lists": True
    }
    
    attachments: Dict[str, Any] = {
        "enabled": True,
        "types": ["pdf", "docx", "xlsx", "png"],
        "max_per_page": 3,
        "total_size_limit_mb": 100
    }
    
    search_optimization: Dict[str, Any] = {
        "semantic_pairs": True,
        "cql_friendly_metadata": True,
        "hybrid_optimized_content": True
    }
    
    safety: Dict[str, Any] = {
        "dry_run": False,
        "confirm_before_create": True,
        "enable_tracking": True,
        "auto_cleanup_on_error": True
    }

class SafetyManager:
    """Manages safety checks for dummy data operations"""
    
    def __init__(self, prefix: str = "TESTDUM"):
        self.prefix = prefix
    
    def is_safe_to_delete(self, title: str) -> bool:
        """Check if item is safe to delete based on prefix"""
        return title.startswith(self.prefix)

def fetch_spaces_by_pattern(client: ConfluenceClient, pattern: str = "TESTDUM") -> List[Dict[str, Any]]:
    """Fetch all spaces from Confluence that match the given pattern
    
    Args:
        client: Confluence client instance
        pattern: Pattern to match space keys against (default: TESTDUM)
    
    Returns:
        List of spaces matching the pattern
    """
    console.print(f"[yellow]Fetching all spaces matching pattern '{pattern}'...[/yellow]")
    
    matching_spaces = []
    start = 0
    limit = 100
    
    while True:
        try:
            response = client.atlassian_api.get_all_spaces(
                start=start, 
                limit=limit
            )
            
            if not response or 'results' not in response:
                break
            
            spaces = response.get('results', [])

            for space in spaces:
                space_key = space.get('key', '')

                if pattern in space_key:
                    matching_spaces.append(space)
                    console.print(f"  • Found: {space_key} - {space.get('name', 'Unknown')}")
            
            if 'next' not in response.get('_links', {}) or len(spaces) < limit:
                break
                
            start += limit
            
        except Exception as e:
            console.print(f"[red]Error fetching spaces: {e}[/red]")
            break
    
    console.print(f"[green]Found {len(matching_spaces)} spaces matching pattern '{pattern}'[/green]")
    return matching_spaces

def check_existing_data(client: ConfluenceClient, pattern: str = "TESTDUM") -> bool:
    """Check if valid test data already exists
    
    Args:
        client: Confluence client instance
        pattern: Pattern to match for existing spaces
    
    Returns:
        True if reusable data found, False otherwise
    """
    console.print("\n[bold yellow]Checking existing test data...[/bold yellow]")

    matching_spaces = fetch_spaces_by_pattern(client, pattern)
    
    if matching_spaces:
        console.print(f"\n[green]Found {len(matching_spaces)} existing test space(s) that can be reused.[/green]")
        console.print("\n[bold]Existing test data summary:[/bold]")

        for space in matching_spaces[:10]:
            console.print(f"  • {space.get('key', '')} - {space.get('name', 'Unknown')}")
        
        if len(matching_spaces) > 10:
            console.print(f"  ... and {len(matching_spaces) - 10} more")
        
        return True
    
    return False

@app.command()
def create(
    config_file: Path = typer.Option(
        Path("scripts/config/dummy_data_config.yaml"),
        "--config",
        "-c",
        help="Configuration file path"
    ),
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        help="Comma-separated list of categories to generate"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without actually creating data"
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help="Skip confirmation prompt"
    ),
    recreate: bool = typer.Option(
        False,
        "--recreate",
        help="Force regeneration of data (cleanup existing first)"
    ),
    reuse_if_exists: bool = typer.Option(
        True,
        "--reuse-if-exists/--no-reuse-if-exists",
        help="Reuse existing test data if found (default: True)"
    ),
    use_real_data: bool = typer.Option(
        True,
        "--real-data/--dummy-data",
        help="Use real collected data instead of dummy content (default: True)"
    )
):
    """Generate data in Confluence using real or dummy content"""
    title = "Confluence Real Data Generator" if use_real_data else "Confluence Dummy Data Generator"
    console.print(f"\n[bold blue]{title}[/bold blue]\n")

    config = DummyDataConfig()
    if config_file.exists():
        try:
            import yaml
            with open(config_file, "r") as f:
                custom_config = yaml.safe_load(f)
                generation_config = custom_config.get("generation", {})
                
                for key, value in generation_config.items():
                    if hasattr(config, key):
                        if isinstance(getattr(config, key), dict) and isinstance(value, dict):
                            existing_dict = getattr(config, key).copy()
                            existing_dict.update(value)
                            setattr(config, key, existing_dict)
                        else:
                            setattr(config, key, value)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load config file: {e}[/yellow]")
            console.print("[yellow]Using default configuration[/yellow]")

    if dry_run:
        config.safety["dry_run"] = True

    if categories:
        category_list = [c.strip() for c in categories.split(",")]
        for space in config.spaces:
            space["categories"] = [c for c in space["categories"] if c in category_list]

    safety_manager = SafetyManager(prefix=config.prefix)

    try:
        client = ConfluenceClient(confluence_config)
        console.print("[green]✓ Connected to Confluence[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to connect to Confluence: {e}[/red]")
        return

    if reuse_if_exists and not recreate:
        if check_existing_data(client, pattern=config.prefix):
            console.print("\n[bold green]✅ Existing test data can be reused![/bold green]")
            console.print("\nTo generate new data anyway, use one of these options:")
            console.print("  • Add --recreate flag to cleanup and regenerate")
            console.print("  • Add --no-reuse-if-exists flag to generate alongside existing data")
            console.print("  • Run 'cleanup' command first to remove existing data")
            return

    if recreate:
        console.print("\n[yellow]Recreate flag set - cleaning up existing data first...[/yellow]")

        cleanup_by_pattern(pattern=config.prefix, skip_confirmation=True)
        console.print("[green]✓ Cleanup completed[/green]\n")

    timestamp = datetime.now().strftime("%m%d%H%M")

    if not no_confirm and not config.safety["dry_run"]:
        console.print("\n[bold]Generation Summary:[/bold]")
        console.print(f"• Mode: {'REAL DATA' if use_real_data else 'DUMMY DATA'}")
        console.print(f"• Prefix: {config.prefix}-{timestamp}")
        console.print(f"• Spaces: {len(config.spaces)}")
        total_pages = sum(space["pages_count"] for space in config.spaces)
        console.print(f"• Total pages: {total_pages}")
        console.print(f"• Attachments enabled: {config.attachments['enabled']}")
        
        if not typer.confirm("\nProceed with generation?"):
            console.print("[yellow]Generation cancelled[/yellow]")
            return
    
    if config.safety["dry_run"]:
        console.print("\n[yellow]DRY RUN MODE - No data will be created[/yellow]\n")
    
    if use_real_data:
        console.print("[yellow]Using REAL DATA mode - content will be sourced from collected documentation[/yellow]\n")
        # Check if real data is available
        real_data_index = Path("config/real_data/content_index.json")
        if real_data_index.exists():
            with open(real_data_index, 'r') as f:
                index_data = json.load(f)
                if index_data.get('total_entries', 0) == 0:
                    console.print("[red]No real data found! Run 'python real_data_collector.py collect' first.[/red]")
                    return
                console.print(f"[green]Found {index_data['total_entries']} real content entries[/green]\n")
        else:
            console.print("[red]Real data index not found! Run 'python real_data_collector.py collect' first.[/red]")
            return

    # Import from renamed modules to avoid conflicts with directories
    from real_content_generators import ContentGenerator, RealDataContentGenerator, DummyContentGenerator
    from real_attachment_generators import AttachmentGenerator

    content_gen = RealDataContentGenerator(client, config) if use_real_data else ContentGenerator(client, config)
    attachment_gen = AttachmentGenerator(client, config) if config.attachments["enabled"] else None

    stats = {
        "spaces_created": 0,
        "pages_created": 0,
        "attachments_created": 0,
        "errors": 0,
        "start_time": time.time()
    }
    
    try:
        for space_config in config.spaces:
            space_key = f"{config.prefix}{space_config['key']}{timestamp}"
            space_name = f"{config.prefix} - {space_config['name']} ({timestamp})"
            
            console.print(f"\n[bold]Creating space: {space_key}[/bold]")
            
            if not config.safety["dry_run"]:
                try:
                    space = content_gen.create_space(space_key, space_name)
                    stats["spaces_created"] += 1

                    pages_to_create = space_config["pages_count"]
                    console.print(f"Generating {pages_to_create} pages...")
                    
                    for i in track(range(pages_to_create), description="Creating pages"):
                        category = random.choice(space_config["categories"])
                        page_data = content_gen.generate_page_content(category, space_config["key"])
                        
                        page = content_gen.create_page(
                            space_key=space_key,
                            title=page_data["title"],
                            content=page_data["content"],
                            labels=page_data.get("labels", [])
                        )
                        
                        if page:
                            stats["pages_created"] += 1

                            if attachment_gen and random.random() < 0.3:
                                num_attachments = random.randint(1, config.attachments["max_per_page"])
                                console.print(f"[dim]Creating {num_attachments} attachment(s) for page {page['title']}...[/dim]")
                                for j in range(num_attachments):
                                    try:
                                        if attachment_gen.create_attachment(page["id"], category):
                                            stats["attachments_created"] += 1
                                            console.print(f"[green]  ✓ Attachment {j+1}/{num_attachments} uploaded successfully[/green]")
                                        else:
                                            console.print(f"[red]  ✗ Failed to upload attachment {j+1}/{num_attachments}[/red]")
                                    except Exception as e:
                                        console.print(f"[red]  ✗ Error uploading attachment {j+1}/{num_attachments}: {e}[/red]")
                                        logger.error(f"Attachment upload error: {e}", exc_info=True)

                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Error creating space {space_key}: {e}")
                    stats["errors"] += 1
                    
                    if config.safety["auto_cleanup_on_error"]:
                        console.print("[yellow]Auto-cleanup triggered due to error[/yellow]")
                        cleanup_space(space_key, force=True)
            else:

                console.print(f"[dim]Would create space: {space_key} with {space_config['pages_count']} pages[/dim]")
                stats["spaces_created"] += 1
                stats["pages_created"] += space_config["pages_count"]
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted by user[/yellow]")
    
    finally:

        duration = time.time() - stats["start_time"]
        console.print("\n[bold]Generation Summary:[/bold]")
        console.print(f"• Duration: {duration:.1f} seconds")
        console.print(f"• Spaces created: {stats['spaces_created']}")
        console.print(f"• Pages created: {stats['pages_created']}")
        console.print(f"• Attachments created: {stats['attachments_created']}")
        console.print(f"• Errors: {stats['errors']}")

@app.command()
def cleanup(
    space_key: Optional[str] = typer.Option(
        None,
        "--space-key",
        "-s",
        help="Specific space key to clean up"
    ),
    pattern: Optional[str] = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Pattern to match space keys (e.g., 'TESTDUM')"
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Skip confirmation prompt"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force cleanup even without tracking"
    )
):
    """Clean up generated dummy data
    
    Examples:
        # Clean all TESTDUM spaces
        python generate_dummy_data.py cleanup --pattern TESTDUM --confirm

        python generate_dummy_data.py cleanup --space-key TESTDUMTECH01011234
        
        # Clean all tracked data
        python generate_dummy_data.py cleanup --confirm
    """
    safety_manager = SafetyManager()
    
    if pattern:

        cleanup_by_pattern(pattern=pattern, skip_confirmation=confirm)
    elif space_key:

        if not safety_manager.is_safe_to_delete(space_key) and not force:
            console.print(f"[red]Space '{space_key}' does not match safety prefix[/red]")
            console.print("[yellow]Use --force to override safety check[/yellow]")
            return
        
        cleanup_space(space_key, confirm, force)
    else:

        console.print("[yellow]Please specify what to clean up:[/yellow]")
        console.print("  • Use --pattern to clean up spaces by pattern (e.g., --pattern TESTDUM)")
        console.print("  • Use --space-key to clean up a specific space")
        console.print("\nExample: python generate_dummy_data.py cleanup --pattern TESTDUM --confirm")

def cleanup_space(space_key: str, confirm: bool = True, force: bool = False):
    """Clean up a specific space"""
    try:
        client = ConfluenceClient(confluence_config)
        safety_manager = SafetyManager()
        
        console.print(f"[bold]Cleaning up space: {space_key}[/bold]")
        console.print(f"[yellow]Note: Due to Confluence API limitations, spaces cannot be completely deleted.[/yellow]")
        console.print(f"[yellow]We will delete all content and attempt to archive the space.[/yellow]")

        try:
            console.print(f"\n[yellow]Fetching pages from space {space_key}...[/yellow]")
            all_pages = []
            start = 0
            limit = 50
            
            while True:
                try:
                    response = client.atlassian_api.get_all_pages_from_space(
                        space_key, start=start, limit=limit, status=None
                    )
                    if not response:
                        break
                        
                    if hasattr(response, '__iter__') and hasattr(response, '__len__') and not hasattr(response, 'get'):

                        pages = response
                    elif hasattr(response, 'get') and 'results' in response:

                        pages = response.get('results', [])
                    else:
                        pages = []
                    
                    if not pages:
                        break
                        
                    all_pages.extend(pages)
                    
                    if len(pages) < limit:
                        break
                    start += limit
                    
                except Exception as e:
                    console.print(f"[red]Error fetching pages: {e}[/red]")
                    import traceback
                    traceback.print_exc()
                    break
            
            console.print(f"[yellow]Found {len(all_pages)} pages to delete[/yellow]")

            deleted_count = 0
            for page in track(all_pages, description=f"Deleting pages from {space_key}"):
                try:
                    page_id = page.get('id') if isinstance(page, dict) else getattr(page, 'id', None)
                    if page_id:

                        client.atlassian_api.remove_page(page_id, status=None, recursive=True)
                        deleted_count += 1

                        time.sleep(0.1)
                except Exception as e:
                    error_msg = str(e).lower()

                    if 'not found' not in error_msg and '404' not in error_msg:
                        console.print(f"[red]Failed to delete page {page_id}: {e}[/red]")
            
            console.print(f"[green]✓ Deleted {deleted_count} pages from space '{space_key}'[/green]")
            
        except Exception as e:
            console.print(f"[red]Failed to clean up pages: {e}[/red]")
        
        # Try to remove trashed contents
        try:
            console.print(f"\n[yellow]Removing trashed contents from space {space_key}...[/yellow]")
            client.atlassian_api.remove_trashed_contents_by_space(space_key)
            console.print(f"[green]✓ Trashed contents removed from space '{space_key}'[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not remove trashed contents: {e}[/yellow]")

        try:
            console.print(f"\n[yellow]Attempting to delete space {space_key} via REST API...[/yellow]")

            response = client.atlassian_api.delete(f"rest/api/space/{space_key}")
            console.print(f"[green]✓ Space '{space_key}' deleted successfully![/green]")
        except Exception as e:

            try:
                console.print(f"[yellow]Delete failed, attempting to archive space {space_key}...[/yellow]")
                client.atlassian_api.archive_space(space_key)
                console.print(f"[green]✓ Space '{space_key}' archived[/green]")
            except Exception as e2:

                console.print(f"[yellow]Could not delete or archive space: {e}[/yellow]")
                console.print(f"[yellow]This may be due to API limitations or permissions.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error cleaning up space: {e}[/red]")

def cleanup_by_pattern(pattern: str = "TESTDUM", skip_confirmation: bool = False):
    """Clean up all spaces matching the given pattern
    
    Args:
        pattern: Pattern to match space keys against (default: TESTDUM)
        skip_confirmation: Skip confirmation prompt
    """
    try:
        client = ConfluenceClient(confluence_config)
        console.print("[green]✓ Connected to Confluence[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Failed to connect to Confluence: {e}[/red]")
        return

    matching_spaces = fetch_spaces_by_pattern(client, pattern)
    
    if not matching_spaces:
        console.print(f"[yellow]No spaces found matching pattern '{pattern}'[/yellow]")
        return
    
    console.print(f"\n[bold]Found {len(matching_spaces)} spaces to clean up:[/bold]")
    for space in matching_spaces[:10]:  # Show first 10
        console.print(f"  • {space['key']} - {space.get('name', 'Unknown')}")
    if len(matching_spaces) > 10:
        console.print(f"  ... and {len(matching_spaces) - 10} more")

    if not skip_confirmation:
        if not typer.confirm(f"\nDelete all {len(matching_spaces)} spaces matching pattern '{pattern}'?"):
            console.print("[yellow]Cleanup cancelled[/yellow]")
            return
    
    console.print(f"\n[bold]Starting cleanup of {len(matching_spaces)} spaces...[/bold]")
    
    total_deleted_pages = 0
    total_deleted_attachments = 0
    cleaned_spaces = 0
    failed_spaces = []

    for space in track(matching_spaces, description="Cleaning up spaces"):
        space_key = space['key']
        space_name = space.get('name', 'Unknown')
        
        try:
            console.print(f"\n[bold]Processing space: {space_key} - {space_name}[/bold]")

            all_pages = []
            start = 0
            limit = 50
            
            while True:
                try:
                    response = client.atlassian_api.get_all_pages_from_space(
                        space_key, start=start, limit=limit, status=None
                    )
                    if not response:
                        break
                        
                    if hasattr(response, '__iter__') and hasattr(response, '__len__') and not hasattr(response, 'get'):
                        pages = response
                    elif hasattr(response, 'get') and 'results' in response:
                        pages = response.get('results', [])
                    else:
                        pages = []
                    
                    if not pages:
                        break
                        
                    all_pages.extend(pages)
                    
                    if len(pages) < limit:
                        break
                    start += limit
                    
                except Exception as e:
                    console.print(f"[red]Error fetching pages from {space_key}: {e}[/red]")
                    break

            deleted_pages = 0
            for page in all_pages:
                try:
                    page_id = page.get('id') if isinstance(page, dict) else getattr(page, 'id', None)
                    if page_id:

                        try:
                            attachments = client.atlassian_api.get_attachments_from_content(page_id)
                            if attachments and 'results' in attachments:
                                for attachment in attachments['results']:
                                    try:
                                        att_id = attachment.get('id')
                                        if att_id:
                                            client.atlassian_api.delete_attachment_by_id(att_id, version=None)
                                            total_deleted_attachments += 1
                                    except:
                                        pass
                        except:
                            pass

                        client.atlassian_api.remove_page(page_id, status=None, recursive=True)
                        deleted_pages += 1
                        time.sleep(0.05)
                        
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'not found' not in error_msg and '404' not in error_msg:
                        logger.error(f"Failed to delete page {page_id}: {e}")
            
            total_deleted_pages += deleted_pages
            console.print(f"[green]✓ Deleted {deleted_pages} pages from space {space_key}[/green]")
            
            try:
                client.atlassian_api.remove_trashed_contents_by_space(space_key)
                console.print(f"[green]✓ Trashed contents removed from space {space_key}[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not remove trashed contents: {e}[/yellow]")

            try:
                response = client.atlassian_api.delete(f"rest/api/space/{space_key}")
                console.print(f"[green]✓ Space '{space_key}' deleted successfully![/green]")
                cleaned_spaces += 1
            except Exception as e:

                try:
                    client.atlassian_api.archive_space(space_key)
                    console.print(f"[green]✓ Space '{space_key}' archived[/green]")
                    cleaned_spaces += 1
                except Exception as e2:
                    console.print(f"[yellow]Could not delete or archive space {space_key}: {e}[/yellow]")

                    if deleted_pages > 0:
                        cleaned_spaces += 1
                    else:
                        failed_spaces.append(space_key)
            
        except Exception as e:
            logger.error(f"Failed to clean space {space_key}: {e}")
            console.print(f"[red]Error cleaning space {space_key}: {e}[/red]")
            failed_spaces.append(space_key)
    
    console.print(f"\n[bold green]✓ Cleanup completed![/bold green]")
    console.print(f"  • Spaces cleaned: {cleaned_spaces}/{len(matching_spaces)}")
    console.print(f"  • Pages deleted: {total_deleted_pages}")
    console.print(f"  • Attachments deleted: {total_deleted_attachments}")
    
    if failed_spaces:
        console.print(f"\n[yellow]⚠️ {len(failed_spaces)} space(s) could not be cleaned:[/yellow]")
        for space_key in failed_spaces[:5]:
            console.print(f"  • {space_key}")
        if len(failed_spaces) > 5:
            console.print(f"  ... and {len(failed_spaces) - 5} more")
    
    if cleaned_spaces < len(matching_spaces):
        console.print(f"\n[yellow]⚠️ Due to Confluence API limitations, some spaces may not be completely deleted.[/yellow]")
        console.print(f"[yellow]Empty spaces will remain but all content has been removed.[/yellow]")

if __name__ == "__main__":
    app()