"""Wire settings to concrete sources. The only place that knows about API keys."""

import logging

from copilot.config import Settings
from copilot.data.entsoe import EntsoeSource, real_client
from copilot.data.fingrid import FingridClient
from copilot.data.fingrid_source import FingridSource

log = logging.getLogger(__name__)


def entsoe_source(settings: Settings) -> EntsoeSource | None:
    if not settings.entsoe_api_key:
        log.warning("ENTSOE_API_KEY missing: prices, load, generation, flows unavailable")
        return None
    return EntsoeSource(real_client(settings.entsoe_api_key), cache_dir=settings.cache_dir)


def fingrid_source(settings: Settings) -> FingridSource | None:
    if not settings.fingrid_api_key:
        log.warning("FINGRID_API_KEY missing: real-time wind/nuclear/imbalance unavailable")
        return None
    return FingridSource(FingridClient(settings.fingrid_api_key), cache_dir=settings.cache_dir)
