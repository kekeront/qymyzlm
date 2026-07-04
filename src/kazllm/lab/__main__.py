"""Entry point: ``python -m kazllm.lab`` launches the testing lab server.

Host and port default to 127.0.0.1:8000 and are overridable via the
``QYMYZLM_HOST`` / ``QYMYZLM_PORT`` environment variables.
"""

import os

import uvicorn


def main() -> None:
    """Run the FastAPI app under uvicorn and print the URL it's serving on."""
    host = os.environ.get("QYMYZLM_HOST", "127.0.0.1")
    port = int(os.environ.get("QYMYZLM_PORT", "8000"))
    print(f"QymyzLM testing lab: http://{host}:{port}")
    uvicorn.run("kazllm.lab.server:app", host=host, port=port)


if __name__ == "__main__":
    main()
