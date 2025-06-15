#!/usr/bin/env python3
"""
Script to migrate content from one branch to another in a Hugging Face repository.

This script uses the Hugging Face Hub Python SDK to download all files from a source branch
and upload them to a target branch, effectively migrating the content.
"""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, login, snapshot_download


def setup_logging() -> None:
    """Set up logging configuration."""
    formatter = logging.Formatter(
        '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        '%m-%d %H:%M:%S'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


def download_branch_content(
    repo_id: str, 
    source_branch: str,
    local_dir: Path,
    token: Optional[str] = None
) -> bool:
    """
    Download all content from a specific branch of a Hugging Face repository.
    
    Args:
        repo_id: The repository ID (e.g., "a6047425318/smolvla-whiteboard-and-bike-light-v4")
        source_branch: The branch to download from
        local_dir: Local directory to download content to
        token: Hugging Face token for authentication
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logging.info(f"Downloading content from {repo_id}:{source_branch} to {local_dir}")
        
        snapshot_download(
            repo_id=repo_id,
            revision=source_branch,
            local_dir=str(local_dir),
            token=token,
            repo_type="model",  # Change to "dataset" or "space" if needed
            resume_download=True
        )
        
        logging.info("Download completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to download branch content: {e}")
        return False


def upload_to_branch(
    repo_id: str,
    target_branch: str,
    local_dir: Path,
    commit_message: str,
    token: Optional[str] = None
) -> bool:
    """
    Upload all content from local directory to a specific branch.
    
    Args:
        repo_id: The repository ID
        target_branch: The branch to upload to
        local_dir: Local directory containing files to upload
        commit_message: Commit message for the upload
        token: Hugging Face token for authentication
        
    Returns:
        True if successful, False otherwise
    """
    try:
        api = HfApi(token=token)
        
        logging.info(f"Uploading content to {repo_id}:{target_branch}")
        
        # Get all files to upload (excluding .git and other hidden files)
        files_to_upload = []
        for file_path in local_dir.rglob("*"):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                relative_path = file_path.relative_to(local_dir)
                files_to_upload.append(str(relative_path))
        
        if not files_to_upload:
            logging.warning("No files found to upload")
            return False
        
        logging.info(f"Found {len(files_to_upload)} files to upload")
        
        # Upload folder to the repository
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            revision=target_branch,
            commit_message=commit_message,
            repo_type="model",  # Change to "dataset" or "space" if needed
            delete_patterns=["*"],  # Delete all existing files first
        )
        
        logging.info("Upload completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to upload to branch: {e}")
        return False


def migrate_branch_content(
    repo_id: str,
    source_branch: str,
    target_branch: str,
    commit_message: str,
    token: Optional[str] = None
) -> bool:
    """
    Migrate content from source branch to target branch.
    
    Args:
        repo_id: The repository ID
        source_branch: The branch to migrate from
        target_branch: The branch to migrate to
        commit_message: Commit message for the migration
        token: Hugging Face token for authentication
        
    Returns:
        True if successful, False otherwise
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        download_dir = temp_path / "download"
        download_dir.mkdir()
        
        # Download content from source branch
        if not download_branch_content(repo_id, source_branch, download_dir, token):
            return False
        
        # Upload content to target branch
        if not upload_to_branch(repo_id, target_branch, download_dir, commit_message, token):
            return False
        
        return True


def main() -> None:
    """Main function to handle command line arguments and execute migration."""
    parser = argparse.ArgumentParser(
        description="Migrate content from one branch to another in a Hugging Face repository"
    )
    parser.add_argument(
        '-r', '--repo-id',
        required=True,
        help="Repository ID (e.g., 'a6047425318/smolvla-whiteboard-and-bike-light-v4')"
    )
    parser.add_argument(
        '-s', '--source-branch',
        default="step-7000",
        help="Source branch to migrate from (default: step-7000)"
    )
    parser.add_argument(
        '-t', '--target-branch', 
        default="main",
        help="Target branch to migrate to (default: main)"
    )
    parser.add_argument(
        '-m', '--message',
        default="Migrate content from source branch",
        help="Commit message for the migration"
    )
    parser.add_argument(
        '--token',
        help="Hugging Face token for authentication (if not provided, will try to use cached token)"
    )
    parser.add_argument(
        '--login',
        action='store_true',
        help="Force login to Hugging Face"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Handle authentication
    if args.login:
        logging.info("Logging in to Hugging Face...")
        login()  # This will prompt for token if needed
    
    # Perform migration
    logging.info(f"Starting migration from {args.repo_id}:{args.source_branch} to {args.target_branch}")
    
    success = migrate_branch_content(
        repo_id=args.repo_id,
        source_branch=args.source_branch,
        target_branch=args.target_branch,
        commit_message=args.message,
        token=args.token
    )
    
    if success:
        logging.info("Migration completed successfully!")
        print(f"✅ Successfully migrated content from {args.source_branch} to {args.target_branch}")
    else:
        logging.error("Migration failed!")
        print(f"❌ Failed to migrate content from {args.source_branch} to {args.target_branch}")
        exit(1)


if __name__ == "__main__":
    main() 