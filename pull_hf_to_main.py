#!/usr/bin/env python3
"""
Script to pull a specific branch from Hugging Face and push to main.

This script clones the HF repository to a temporary directory, extracts the specified branch,
and then pushes it to your repository's main branch without affecting your current working directory.
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
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


def get_current_repo_info() -> tuple[Optional[str], Optional[str]]:
    """Get current repository's origin URL and main branch name."""
    # Get origin URL
    success, origin_url, _ = run_git_command(["git", "remote", "get-url", "origin"])
    if not success:
        logging.error("Could not get origin URL. Make sure you're in a git repository.")
        return None, None
    
    # Determine main branch name
    success, _, _ = run_git_command(["git", "show-ref", "--verify", "--quiet", "refs/heads/main"])
    if success:
        main_branch = "main"
    else:
        success, _, _ = run_git_command(["git", "show-ref", "--verify", "--quiet", "refs/heads/master"])
        if success:
            main_branch = "master"
        else:
            logging.error("Could not find main or master branch")
            return None, None
    
    return origin_url, main_branch


def clone_hf_repo(repo_path: str, temp_dir: Path) -> Optional[Path]:
    """
    Clone the Hugging Face repository to a temporary directory.
    
    Args:
        repo_path: The Hugging Face repository path
        temp_dir: Temporary directory to clone into
        
    Returns:
        Path to the cloned repository or None if failed
    """
    hf_url = f"https://huggingface.co/{repo_path}"
    clone_path = temp_dir / "hf_repo"
    
    logging.info(f"Cloning {hf_url} to {clone_path}")
    success, _, stderr = run_git_command(["git", "clone", hf_url, str(clone_path)])
    if not success:
        logging.error(f"Failed to clone repository: {stderr}")
        return None
    
    return clone_path


def checkout_branch(repo_path: Path, branch_name: str) -> bool:
    """
    Checkout the specified branch in the cloned repository.
    
    Args:
        repo_path: Path to the repository
        branch_name: Name of the branch to checkout
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Checking out branch '{branch_name}'")
    success, _, stderr = run_git_command(["git", "checkout", branch_name], cwd=repo_path)
    if not success:
        logging.error(f"Failed to checkout branch: {stderr}")
        return False
    
    return True


def setup_target_repo(temp_dir: Path, origin_url: str, main_branch: str) -> Optional[Path]:
    """
    Clone the target repository and set it up for pushing.
    
    Args:
        temp_dir: Temporary directory
        origin_url: URL of the target repository
        main_branch: Name of the main branch
        
    Returns:
        Path to the target repository or None if failed
    """
    target_path = temp_dir / "target_repo"
    
    logging.info(f"Cloning target repository to {target_path}")
    success, _, stderr = run_git_command(["git", "clone", origin_url, str(target_path)])
    if not success:
        logging.error(f"Failed to clone target repository: {stderr}")
        return None
    
    # Checkout main branch
    success, _, stderr = run_git_command(["git", "checkout", main_branch], cwd=target_path)
    if not success:
        logging.error(f"Failed to checkout {main_branch}: {stderr}")
        return None
    
    return target_path


def copy_repo_contents(source_path: Path, target_path: Path) -> bool:
    """
    Copy contents from source repository to target repository.
    
    Args:
        source_path: Path to source repository
        target_path: Path to target repository
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Copying contents from {source_path} to {target_path}")
    
    try:
        # Remove all files except .git directory
        for item in target_path.iterdir():
            if item.name != '.git':
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        
        # Copy all files from source except .git directory
        for item in source_path.iterdir():
            if item.name != '.git':
                target_item = target_path / item.name
                if item.is_dir():
                    shutil.copytree(item, target_item)
                else:
                    shutil.copy2(item, target_item)
        
        logging.info("Contents copied successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to copy contents: {e}")
        return False


def commit_and_push(repo_path: Path, main_branch: str, commit_message: str) -> bool:
    """
    Commit changes and push to main branch.
    
    Args:
        repo_path: Path to the repository
        main_branch: Name of the main branch
        commit_message: Commit message
        
    Returns:
        True if successful, False otherwise
    """
    # Add all changes
    success, _, stderr = run_git_command(["git", "add", "."], cwd=repo_path)
    if not success:
        logging.error(f"Failed to add changes: {stderr}")
        return False
    
    # Check if there are changes to commit
    success, stdout, _ = run_git_command(["git", "status", "--porcelain"], cwd=repo_path)
    if not success or not stdout:
        logging.info("No changes to commit")
        return True
    
    # Commit changes
    success, _, stderr = run_git_command(["git", "commit", "-m", commit_message], cwd=repo_path)
    if not success:
        logging.error(f"Failed to commit changes: {stderr}")
        return False
    
    # Push to main branch
    logging.info(f"Pushing to {main_branch}")
    success, _, stderr = run_git_command(["git", "push", "origin", main_branch], cwd=repo_path)
    if not success:
        logging.error(f"Failed to push to {main_branch}: {stderr}")
        return False
    
    logging.info("Successfully pushed to main!")
    return True


def main() -> None:
    """Main function to orchestrate the operation."""
    parser = argparse.ArgumentParser(
        description="Pull a branch from Hugging Face and push to your repository's main branch"
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
        "-m", "--message",
        default="Update from Hugging Face model",
        help="Commit message (default: 'Update from Hugging Face model')"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info("Starting Hugging Face to main branch sync")
    logging.info(f"HF Repository: {args.repo}")
    logging.info(f"Branch: {args.branch}")
    logging.info(f"Commit message: {args.message}")
    
    # Get current repository info
    origin_url, main_branch = get_current_repo_info()
    if not origin_url or not main_branch:
        sys.exit(1)
    
    logging.info(f"Target repository: {origin_url}")
    logging.info(f"Main branch: {main_branch}")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        logging.info(f"Using temporary directory: {temp_dir}")
        
        # Clone HF repository
        hf_repo_path = clone_hf_repo(args.repo, temp_dir)
        if not hf_repo_path:
            sys.exit(1)
        
        # Checkout the specified branch
        if not checkout_branch(hf_repo_path, args.branch):
            sys.exit(1)
        
        # Clone target repository
        target_repo_path = setup_target_repo(temp_dir, origin_url, main_branch)
        if not target_repo_path:
            sys.exit(1)
        
        # Copy contents
        if not copy_repo_contents(hf_repo_path, target_repo_path):
            sys.exit(1)
        
        # Commit and push
        if not commit_and_push(target_repo_path, main_branch, args.message):
            sys.exit(1)
    
    logging.info("Script completed successfully!")
    logging.info(f"The {args.branch} branch from {args.repo} has been pushed to your {main_branch} branch")


if __name__ == "__main__":
    main() 