"""QymyzEmbed: Kazakh text-embedding models.

v0: Less-is-More fine-tune of intfloat/multilingual-e5-base (arXiv 2603.22290).
All evaluation numbers come from evallab/ runners (see qymyz_embed.evaluate).
"""

from qymyz_embed.prefixes import E5_PROMPTS, PASSAGE_PREFIX, QUERY_PREFIX

__all__ = ["E5_PROMPTS", "PASSAGE_PREFIX", "QUERY_PREFIX", "__version__"]
__version__ = "0.1.0"
