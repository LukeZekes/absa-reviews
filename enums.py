from enum import IntEnum

class TermClasses(IntEnum):
  INVALID = 0
  ASPECT = 1
  OPINION = 2

class SentimentClasses(IntEnum):
  NEGATIVE = 0
  INVALID = 1
  NEUTRAL = 2
  POSITIVE = 3
