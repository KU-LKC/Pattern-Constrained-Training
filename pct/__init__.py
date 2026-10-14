"""Pattern-constrained training (PCT): truth-table-driven, certified error-free weight
projection for approximate compute-in-memory adder trees."""

from .compiler import Certificate, CompiledCompressor, compile_compressor, minimal_error_patterns
from .convert import ConvertPolicy, ConvertReport, convert_model
from .layout import PatternLayout, compile_layout
from .spec import CompressorDef, HWSpec, load_compressor, parse_truth_table

__all__ = [
    "Certificate",
    "CompiledCompressor",
    "CompressorDef",
    "ConvertPolicy",
    "ConvertReport",
    "HWSpec",
    "PatternLayout",
    "compile_compressor",
    "compile_layout",
    "convert_model",
    "load_compressor",
    "minimal_error_patterns",
    "parse_truth_table",
]
