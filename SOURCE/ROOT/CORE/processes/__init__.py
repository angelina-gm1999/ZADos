"""
ZA-DOS v0.6 — Process modules.

Shared infrastructure consumed by all input-mode pipelines:
  - subject_classifier      7 SubjectCategory classification
  - engine_toolkit           Mode × Subject → EngineTier matrix
  - emotional_landscape      EmotionalPreset configs (M1-M5)
  - context_anchor           ContextAnchorManager + drift detection
  - learning_log             LearningLogPipeline
  - unsolved_buffer          UnsolvedBuffer + LTMM persistence
  - intent_pipeline_optimizer  8 intent → PipelineDepthConfig profiles
"""
