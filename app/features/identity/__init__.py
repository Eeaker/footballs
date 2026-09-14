"""Identity publication, isolation, labels, and avatar selection."""

from .policy import IdentityCatalog, build_display_id, load_identity_catalog

__all__ = ["IdentityCatalog", "build_display_id", "load_identity_catalog"]

