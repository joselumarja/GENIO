"""
FeatureParams.py
Feature extraction methods for audio preprocessing
UC - GIM - Version 1.0
Author: Marina Cordovilla
"""

from dataclasses import dataclass # for data class definition

#
# Class FeatureParams
# It defines the configuration parameters for the feature extraction methods.
#

@dataclass
class FeatureParams:
  """audio sample rate"""
  sample_rate: int = @SAMPLE_RATE@ 
  
  """input window size in samples"""
  window_size_samples: int = @WINDOWS_SIZE_SAMPLES@
  
  """input window hop (shift between consecutive windows) in samples"""
  window_hop_samples: int = @WINDOW_HOP_SAMPLES@
  
  """FFT size"""
  FFT_size: int = @FFT_SIZE@
  
  """Hop length"""
  Hop_length: int = @HOP_LENGTH@
  
  """ Number of bands in the MEL filter bank """
  MEL_band_number: int = @MEL_BAND_NUMBER@
  
  """ MEL filter bank lower band limit"""
  lower_band_limit_hz: float = @LOWER_BAND_LIMIT_HZ@
  
  """ MEL filter bank upper band limit"""
  upper_band_limit_hz: float = @UPPER_BAND_LIMIT_HZ@
  

