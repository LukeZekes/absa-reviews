from enum import IntEnum

class TermClasses(IntEnum):
  INVALID = 0
  ASPECT = 1
  OPINION = 2

class SentimentClasses(IntEnum):
  NEGATIVE = -1
  INVALID = 0
  NEUTRAL = 1
  POSITIVE = 2
