#!/usr/bin/env python3
"""
Script to pull a specific branch from Hugging Face and push to main.

This script pulls the specified branch from the Hugging Face repository
and pushes it to the main branch of the current repository.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional


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


def run_git_command(command: list[str], cwd: Optional[Path] = None) -> tuple[bool, str, str]:
    """
    Run a git command and return success status, stdout, and stderr.
    
    Args:
        command: Git command as list of strings
        cwd: Working directory for the command
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        
        success = result.returncode == 0
        return success, result.stdout.strip(), result.stderr.strip()
        
    except Exception as e:
        logging.error(f"Failed to run git command {' '.join(command)}: {e}")
        return False, "", str(e)


def check_git_status() -> bool:
    """Check if the current directory is a git repository and is clean."""
    # Check if we're in a git repo
    success, _, _ = run_git_command(["git", "rev-parse", "--git-dir"])
    if not success:
        logging.error("Current directory is not a git repository")
        return False
    
    # Check if working tree is clean
    success, stdout, _ = run_git_command(["git", "status", "--porcelain"])
    if not success:
        logging.error("Failed to check git status")
        return False
    
    if stdout:
        logging.error("Working tree is not clean. Please commit or stash changes first.")
        logging.info("Uncommitted changes:")
        for line in stdout.split('\n'):
            logging.info(f"  {line}")
        return False
    
    return True


def add_huggingface_remote(repo_path: str, remote_name: str = "huggingface") -> bool:
    """
    Add Hugging Face repository as a remote.
    
    Args:
        repo_path: The Hugging Face repository path (e.g., "a6047425318/smolvla-whiteboard-and-bike-light-v3")
        remote_name: Name for the remote
        
    Returns:
        True if successful, False otherwise
    """
    hf_url = f"https://huggingface.co/{repo_path}"
    
    # Check if remote already exists
    success, stdout, _ = run_git_command(["git", "remote", "get-url", remote_name])
    if success:
        if stdout == hf_url:
            logging.info(f"Remote '{remote_name}' already exists with correct URL")
            return True
        else:
            logging.info(f"Updating remote '{remote_name}' URL")
            success, _, stderr = run_git_command(["git", "remote", "set-url", remote_name, hf_url])
            if not success:
                logging.error(f"Failed to update remote URL: {stderr}")
                return False
    else:
        # Add new remote
        logging.info(f"Adding remote '{remote_name}' with URL: {hf_url}")
        success, _, stderr = run_git_command(["git", "remote", "add", remote_name, hf_url])
        if not success:
            logging.error(f"Failed to add remote: {stderr}")
            return False
    
    return True


def fetch_remote_branch(remote_name: str, branch_name: str) -> bool:
    """
    Fetch the specified branch from the remote.
    
    Args:
        remote_name: Name of the remote
        branch_name: Name of the branch to fetch
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Fetching branch '{branch_name}' from remote '{remote_name}'...")
    success, _, stderr = run_git_command(["git", "fetch", remote_name, branch_name])
    if not success:
        logging.error(f"Failed to fetch branch: {stderr}")
        return False
    
    logging.info("Fetch completed successfully")
    return True


def checkout_and_push_to_main(remote_name: str, branch_name: str, force_push: bool = False) -> bool:
    """
    Checkout the remote branch and push to main.
    
    Args:
        remote_name: Name of the remote
        branch_name: Name of the branch
        force_push: Whether to force push to main
        
    Returns:
        True if successful, False otherwise
    """
    remote_branch = f"{remote_name}/{branch_name}"
    
    # Create a temporary local branch tracking the remote branch
    temp_branch = f"temp-{branch_name}"
    logging.info(f"Creating temporary branch '{temp_branch}' from '{remote_branch}'")
    
    # Delete temp branch if it exists
    run_git_command(["git", "branch", "-D", temp_branch])
    
    # Create new branch tracking remote
    success, _, stderr = run_git_command(["git", "checkout", "-b", temp_branch, remote_branch])
    if not success:
        logging.error(f"Failed to checkout remote branch: {stderr}")
        return False
    
    # Switch to main branch
    logging.info("Switching to main branch")
    success, _, stderr = run_git_command(["git", "checkout", "main"])
    if not success:
        # Try 'master' if 'main' doesn't exist
        success, _, stderr = run_git_command(["git", "checkout", "master"])
        if not success:
            logging.error(f"Failed to checkout main/master branch: {stderr}")
            return False
        main_branch = "master"
    else:
        main_branch = "main"
    
    # Reset main to the temp branch
    logging.info(f"Resetting {main_branch} to match {temp_branch}")
    success, _, stderr = run_git_command(["git", "reset", "--hard", temp_branch])
    if not success:
        logging.error(f"Failed to reset {main_branch}: {stderr}")
        return False
    
    # Push to origin
    push_command = ["git", "push", "origin", main_branch]
    if force_push:
        push_command.append("--force")
    
    logging.info(f"Pushing to origin/{main_branch}...")
    success, _, stderr = run_git_command(push_command)
    if not success:
        logging.error(f"Failed to push to origin: {stderr}")
        return False
    
    # Clean up temp branch
    logging.info(f"Cleaning up temporary branch '{temp_branch}'")
    run_git_command(["git", "branch", "-D", temp_branch])
    
    logging.info("Successfully pushed to main!")
    return True


def main() -> None:
    """Main function to orchestrate the git operations."""
    parser = argparse.ArgumentParser(
        description="Pull a branch from Hugging Face and push to main"
    )
    parser.add_argument(
        "-r", "--repo",
        default="a6047425318/smolvla-whiteboard-and-bike-light-v3",
        help="Hugging Face repository path (default: a6047425318/smolvla-whiteboard-and-bike-light-v3)"
    )
    parser.add_argument(
        "-b", "--branch",
        default="step-0200",
        help="Branch name to pull (default: step-0200)"
    )
    parser.add_argument(
        "--remote-name",
        default="huggingface",
        help="Name for the Hugging Face remote (default: huggingface)"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force push to main (use with caution)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info("Starting Hugging Face branch pull script")
    logging.info(f"Repository: {args.repo}")
    logging.info(f"Branch: {args.branch}")
    logging.info(f"Remote name: {args.remote_name}")
    logging.info(f"Force push: {args.force}")
    
    # Check git status
    if not check_git_status():
        sys.exit(1)
    
    # Add Hugging Face remote
    if not add_huggingface_remote(args.repo, args.remote_name):
        sys.exit(1)
    
    # Fetch the branch
    if not fetch_remote_branch(args.remote_name, args.branch):
        sys.exit(1)
    
    # Checkout and push to main
    if not checkout_and_push_to_main(args.remote_name, args.branch, args.force):
        sys.exit(1)
    
    logging.info("Script completed successfully!")


if __name__ == "__main__":
    main() 