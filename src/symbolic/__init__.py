"""Symbolic reasoning module for PathVQA.

Components:
  - scene_graph.py: SceneGraph dataclass for fact representation
  - scene_parser.py: Neural module predicting symbolic attributes from visual features
  - query_parser.py: Rule-based question classification into structured queries
  - executor.py: Maps scene logits + query type → answer vocabulary logits
"""
