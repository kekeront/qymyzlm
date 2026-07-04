"""Entry point: ``python -m kazllm.cockpit`` launches the campaign dashboard.

Host is fixed to 127.0.0.1 (read-only local tool, not for network exposure).
Port defaults to 8777 and is overridable via the ``COCKPIT_PORT`` environment
variable.
"""

import os

import uvicorn


def main() -> None:
    """Run the FastAPI app under uvicorn and print the URL it's serving on."""
    host = "127.0.0.1"
    port = int(os.environ.get("COCKPIT_PORT", "8777"))
    print(f"Campaign Mission Control: http://{host}:{port}")
    uvicorn.run("kazllm.cockpit.server:app", host=host, port=port)


if __name__ == "__main__":
    main()
