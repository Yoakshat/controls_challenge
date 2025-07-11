from . import BaseController

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.25
    self.i = 0.1
    self.d = -0.1
    self.error_integral = 0
    self.prev_error = 0
    # self.prev_lataccel = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      # self.prev_lataccel = current_lataccel
      return self.p * error + self.i * self.error_integral + self.d * error_diff