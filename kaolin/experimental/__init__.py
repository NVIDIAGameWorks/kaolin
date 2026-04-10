try:
    from . import newton
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("newton is not installed, install newton==1.0.0 to use kaolin.experimental.newton")

