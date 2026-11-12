"""CLI Utility to link GitHub repository, DagsHub DVC Remote, and DagsHub MLflow."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv


def run_cmd(cmd: list[str], mask_tokens: list[str] | None = None) -> None:
    """Execute a shell command and log it safely without leaking secrets."""
    display_cmd = " ".join(cmd)
    for secret in mask_tokens or []:
        if secret:
            display_cmd = display_cmd.replace(secret, "********")
    print(f"-> Running: {display_cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        print(f"Warning/Error: {result.stderr.strip()}", file=sys.stderr)


def configure_remotes(
    github_url: str | None,
    dagshub_owner: str | None,
    dagshub_repo: str | None,
    dagshub_token: str | None,
    push: bool = False,
) -> None:
    """Configure Git remote, DVC DagsHub remote, and .env file for MLflow."""
    load_dotenv()

    github_url = github_url or os.getenv("GITHUB_REPO_URL", "").strip()
    owner = dagshub_owner or os.getenv("DAGSHUB_REPO_OWNER", "").strip()
    repo = dagshub_repo or os.getenv("DAGSHUB_REPO_NAME", "").strip()
    token = dagshub_token or os.getenv("DAGSHUB_USER_TOKEN", "").strip()

    # Invoke DVC via the active Python interpreter (`python -m dvc`)
    dvc_cmd = [sys.executable, "-m", "dvc"]

    # 1. Configure Git Remote
    if github_url:
        existing = subprocess.run(
            ["git", "remote"], capture_output=True, text=True
        ).stdout.split()
        if "origin" in existing:
            run_cmd(["git", "remote", "set-url", "origin", github_url])
        else:
            run_cmd(["git", "remote", "add", "origin", github_url])
        print(f"[OK] Linked Git remote 'origin' -> {github_url}")

    # 2. Configure DagsHub DVC Remote & MLflow .env
    if owner and repo:
        dvc_url = f"https://dagshub.com/{owner}/{repo}.dvc"
        mlflow_url = f"https://dagshub.com/{owner}/{repo}.mlflow"

        run_cmd([*dvc_cmd, "remote", "add", "-d", "-f", "origin", dvc_url])
        run_cmd([*dvc_cmd, "remote", "modify", "origin", "auth", "basic"])
        run_cmd([*dvc_cmd, "remote", "modify", "origin", "user", owner])

        if token:
            # Stored in .dvc/config.local (automatically git-ignored by DVC)
            run_cmd(
                [*dvc_cmd, "remote", "modify", "--local", "origin", "password", token],
                mask_tokens=[token],
            )

        # Write or update .env for MLflow + DagsHub
        env_lines = [
            f"DAGSHUB_REPO_OWNER={owner}",
            f"DAGSHUB_REPO_NAME={repo}",
            f"MLFLOW_TRACKING_URI={mlflow_url}",
            f"MLFLOW_TRACKING_USERNAME={owner}",
        ]
        if token:
            env_lines.append(f"DAGSHUB_USER_TOKEN={token}")
            env_lines.append(f"MLFLOW_TRACKING_PASSWORD={token}")
        if github_url:
            env_lines.append(f"GITHUB_REPO_URL={github_url}")

        Path(".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        print(f"[OK] Configured DVC remote -> {dvc_url}")
        print(f"[OK] Configured MLflow URI -> {mlflow_url} (saved in .env)")

    # 3. Optional Push to GitHub & DagsHub DVC
    if push:
        if github_url:
            run_cmd(["git", "branch", "-M", "main"])
            run_cmd(["git", "push", "-u", "origin", "main"])
        if owner and repo and token:
            run_cmd([*dvc_cmd, "push", "-r", "origin"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Link project with GitHub, DagsHub DVC, and DagsHub MLflow"
    )
    parser.add_argument("--github-url", type=str, help="GitHub repository URL")
    parser.add_argument("--dagshub-owner", type=str, help="DagsHub username / organization")
    parser.add_argument("--dagshub-repo", type=str, help="DagsHub repository name")
    parser.add_argument("--dagshub-token", type=str, help="DagsHub access token")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push Git commits to GitHub and DVC data/models to DagsHub",
    )
    args = parser.parse_args()

    configure_remotes(
        github_url=args.github_url,
        dagshub_owner=args.dagshub_owner,
        dagshub_repo=args.dagshub_repo,
        dagshub_token=args.dagshub_token,
        push=args.push,
    )

