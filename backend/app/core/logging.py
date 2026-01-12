import logging


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # IMPORTANT: never log raw contract text; log only IDs/status/metadata.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
