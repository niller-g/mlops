import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops"
PYTHON_VERSION = "3.11"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(
        f"python src/{PROJECT_NAME}/data.py data/raw data/processed",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def predict(
    ctx: Context,
    prompt="What are the symptoms of ",
    model_path="models/distilgpt2-finetuned-final",
    max_length=50,
):
    """
    Generate text from the fine-tuned DistilGPT2 model.

    Usage:
        invoke predict --prompt="Explain how viruses differ from bacteria."
                       --model-path="models/distilgpt2-finetuned-final"
                       --max-length=100
    """
    cmd = (
        f"python src/{PROJECT_NAME}/predict.py "
        f'--prompt "{prompt}" '
        f'--model-path "{model_path}" '
        f"--max-length {max_length}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("pip install .[dev]", echo=True, pty=not WINDOWS)
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def setup_data_version_controle(ctx: Context) -> None:
    """If you do not have gcloud installed go to: https://cloud.google.com/sdk/docs/install"""
    ctx.run("gcloud auth login", echo=True, pty=not WINDOWS)
    ctx.run("gcloud auth application-default login", echo=True, pty=not WINDOWS)
    ctx.run("dvc pull", echo=True, pty=not WINDOWS)


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run(
        "mkdocs build --config-file docs/mkdocs.yaml --site-dir build",
        echo=True,
        pty=not WINDOWS,
    )


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def profile_train(ctx, script="src/mlops/train.py", out="profiles/train_profile.prof"):
    """
    Profile the training script with cProfile and visualize with Snakeviz.

    Usage:
        invoke profile-train
        invoke profile-train --script=src/mlops/train.py --out=profiles/custom_train.prof
    """
    # Ensure the output directory exists
    out_dir = os.path.dirname(out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Run cProfile
    cmd_profile = f"python -m cProfile -o {out} -s cumtime {script}"
    ctx.run(cmd_profile, echo=True, pty=not WINDOWS)

    # Launch Snakeviz
    cmd_snakeviz = f"snakeviz {out}"
    ctx.run(cmd_snakeviz, echo=True, pty=not WINDOWS)


@task
def profile_predict(ctx, script="src/mlops/predict.py", out="profiles/predict_profile.prof"):
    """
    Profile the prediction script with cProfile and visualize with Snakeviz.

    Usage:
        invoke profile-predict
        invoke profile-predict --script=src/mlops/predict.py --out=profiles/custom_predict.prof
    """
    # Ensure the output directory exists
    out_dir = os.path.dirname(out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Run cProfile
    cmd_profile = f"python -m cProfile -o {out} -s cumtime {script}"
    ctx.run(cmd_profile, echo=True, pty=not WINDOWS)

    # Launch Snakeviz
    cmd_snakeviz = f"snakeviz {out}"
    ctx.run(cmd_snakeviz, echo=True, pty=not WINDOWS)
