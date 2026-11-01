"""mads.memory.extractors: write-side scaffolding.

Today the only write path is the LLM calling a memory tool explicitly. The
extractors subpackage scaffolds a complementary path: deterministic mining
of agent transcripts for facts the LLM forgot (or never thought) to record.
"""
