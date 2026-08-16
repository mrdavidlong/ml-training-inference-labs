"""Detect the hardware, build the right `vllm serve` command, and run it.

This program does not serve anything itself. It works out what `vllm serve ...`
command suits your machine and chosen goal, prints it, and then runs it. That
design is deliberate: with `--dry-run` you can see the exact command and type it
yourself, so nothing about the setup stays hidden behind a wrapper.

WHAT vLLM IS

Once a model is trained, something has to hold it in memory and answer requests
over the network. vLLM is a server built for that job. It exposes the same HTTP
interface as OpenAI's API, which is why the load tester in this repository can
talk to it without any vendor-specific code.

WHERE THE SETTINGS COME FROM

All the tuning knobs live in configs/serve.toml, not in this file, so trying a
different configuration is a data edit rather than a code change. Three layers
are merged, each overriding the one before:

    [common]              host, port -- shared by everything
        |
        v
    [backend.metal]  or  [backend.cuda]   model name, GPU-specific options
        |
        v
    [profile.learning|latency|throughput]  the tuning goal
        |
        v
    final command:  vllm serve MODEL --host ... --port ... --max-model-len ...

Layering this way means a profile can be swapped without disturbing anything
about the hardware, which is what makes a fair A/B comparison possible.

vLLM server options: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from inference_labs.hardware import BACKENDS, detect_backend


# The repository root, found by climbing out of inference_labs/, so the default
# config resolves no matter which directory the command is run from.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "serve.toml"


def load_config(path: Path) -> dict[str, object]:
    """Read the TOML settings file.

    TOML is a plain-text configuration format, chosen here because Python can
    read it with no extra dependency (`tomllib`, standard since Python 3.11).

    Args:
        path: The .toml file to read, normally configs/serve.toml.

    Returns:
        The file as nested dictionaries. The top-level keys are "common",
        "backend", and "profile".

    Note:
        Opened in binary mode ("rb") because tomllib requires it -- TOML is
        defined as UTF-8, so the parser decodes the bytes itself.
    """
    with path.open("rb") as handle:
        return tomllib.load(handle)


def build_command(
    backend: str,
    profile: str,
    config: dict[str, object],
    model_override: str | None = None,
    host_override: str | None = None,
    port_override: int | None = None,
    api_key: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Assemble the full `vllm serve` command line from the layered settings.

    Args:
        backend: "metal" or "cuda". Selects which [backend.*] section applies.
        profile: "learning", "latency", or "throughput" -- the tuning goal,
            selecting which [profile.*] section applies.
        config: The parsed TOML, as returned by `load_config`.
        model_override: Serve this model instead of the one in the config.
        host_override: Network interface to bind. The config default is
            127.0.0.1, meaning only this machine can connect. Changing it to
            0.0.0.0 exposes the server to your whole network, which has no
            authentication unless you also set an api_key.
        port_override: TCP port to listen on, instead of the config's 8000.
        api_key: If set, clients must send this as a bearer token. Comes from
            the VLLM_API_KEY environment variable by default so it need not be
            typed into a shell history.
        extra_args: Raw flags appended verbatim, for vLLM options this lab does
            not model. Passed through untouched, so they can override anything
            built above by appearing later on the command line.

    Returns:
        The command as a list of strings, ready for `subprocess.run`. A list
        rather than one string means no shell is involved, so a model name with
        an awkward character cannot be misread as shell syntax.

    Raises:
        ValueError: If the backend cannot really serve, or the profile is not
            defined in the config file.
    """
    # "cpu" and "mock" never reach here -- main() diverts them first. This guard
    # catches a programming error rather than a user error.
    if backend not in {"metal", "cuda"}:
        raise ValueError("real serving requires the metal or cuda backend")

    # Copied into new dicts because the code below uses `pop`, and mutating the
    # caller's parsed config would corrupt it for any later call.
    common = dict(config.get("common", {}))
    backend_config = dict(config.get("backend", {}).get(backend, {}))
    profiles = dict(config.get("profile", {}))
    if profile not in profiles:
        raise ValueError(f"unknown profile {profile!r}; choose from {', '.join(sorted(profiles))}")
    profile_config = dict(profiles[profile])

    # These three are positional or specially formatted, so they are pulled out
    # of the dictionaries before the generic flag loop runs. `pop` removes them
    # so they do not get emitted twice.
    #
    # The pops happen on their own lines, before the overrides are applied, and
    # that ordering is load-bearing: writing `model_override or pop(...)` would
    # short-circuit whenever an override was supplied, skip the pop, and leave
    # the key in the dictionary for the loop below to emit a second time. The
    # result was a command carrying both values, such as
    # `vllm serve my-model ... --model the-config-model`.
    config_model = backend_config.pop("model")
    config_host = common.pop("host", "127.0.0.1")
    config_port = common.pop("port", 8000)
    model = model_override or str(config_model)
    host = host_override or str(config_host)
    port = port_override or int(config_port)
    command = ["vllm", "serve", model, "--host", host, "--port", str(port)]

    # The layering from the module docstring. Later dictionaries win on
    # conflicts, so a profile can override a backend default, which can
    # override a common default.
    options = {**common, **backend_config, **profile_config}

    for name, value in options.items():
        # TOML keys use underscores; command-line flags use hyphens. So
        # `max_model_len = 4096` becomes `--max-model-len 4096`.
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            # Booleans are on/off switches that take no value: `true` emits
            # `--enable-prefix-caching`, and `false` emits nothing at all.
            if value:
                command.append(flag)
        elif value is not None:
            command.extend([flag, str(value)])

    if api_key:
        command.extend(["--api-key", api_key])
    command.extend(extra_args or [])
    return command


def main() -> None:
    """Detect hardware, build the serve command, print it, and run it.

    Does not return normally when serving: it hands control to vLLM and then
    exits with whatever code vLLM exited with. With --dry-run it prints the
    command and returns without starting anything.
    """
    parser = argparse.ArgumentParser(
        description="Detect Apple Metal or NVIDIA CUDA and start a real vLLM server"
    )
    parser.add_argument("--backend", choices=BACKENDS, default="auto")
    # Deliberately no `choices=`: the profiles live in the TOML, and hardcoding
    # a copy of the list here would mean a new [profile.*] section could be
    # defined in the config but rejected by the parser before build_command ever
    # saw it. build_command validates against the file and names the ones it
    # found, so the config file stays the single source of truth.
    parser.add_argument(
        "--profile",
        default="learning",
        help="tuning profile to apply; see the [profile.*] sections of configs/serve.toml",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    # Read from the environment so a secret is not typed on the command line,
    # where it would be saved in shell history and visible in process listings.
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    # `action="append"` allows repeating the flag: --extra-arg A --extra-arg B.
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
    args = parser.parse_args()

    hardware = detect_backend(args.backend)
    # Printed to stderr, not stdout, so that piping this program's output
    # captures only the command itself and not the diagnostic line.
    print(f"backend={hardware.backend}: {hardware.reason}", file=sys.stderr)

    if hardware.backend == "cpu":
        # Refusing is kinder than complying. A CPU-only run would take many
        # seconds per response, teaching nothing accurate about latency or
        # throughput, and any measurements taken would be misleading.
        raise SystemExit(
            "No supported accelerator was detected. Use an Apple Silicon Mac, an NVIDIA GPU, "
            "or pass --backend mock only for the test fixture. See docs/06-troubleshooting.md."
        )

    if hardware.backend == "mock":
        # The fake server from mock_server.py. It returns canned text and exists
        # only so the test suite can exercise HTTP streaming and metrics without
        # downloading a model. Never use its timings as real results.
        # `sys.executable` is the current Python interpreter, which guarantees
        # the subprocess runs in the same virtual environment.
        command = [sys.executable, "-m", "inference_labs.mock_server", "--host", args.host or "127.0.0.1", "--port", str(args.port or 8000)]
    else:
        try:
            command = build_command(
                hardware.backend,
                args.profile,
                load_config(args.config),
                model_override=args.model,
                host_override=args.host,
                port_override=args.port,
                api_key=args.api_key,
                extra_args=args.extra_arg,
            )
        except ValueError as error:
            # An unknown --profile is ordinary user error, so it surfaces as a
            # message rather than a traceback. The message already lists the
            # profiles actually present in the config file.
            raise SystemExit(f"{error}. See docs/03-tuning.md.") from error
        # Checked before running, but skipped for --dry-run, so the command can
        # still be previewed on a machine where vLLM is not yet installed.
        if not args.dry_run and shutil.which("vllm") is None:
            raise SystemExit("vllm is not installed in the active environment; follow docs/00-installation.md")

    # `shlex.join` quotes anything that would need it, so the printed line can
    # be copied straight into a terminal and will behave identically.
    print(shlex.join(command))
    if not args.dry_run:
        # Runs vLLM in the foreground and adopts its exit code, so a failure to
        # start is reported rather than swallowed. The server runs until you
        # stop it with Control-C.
        raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
