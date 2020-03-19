import numpy as np

class Bird:
	def __init__(self, initial_position=None,
				 initial_velcoty=None):
		self.position = None
		if initial_position is not None:
			self.position = initial_position
		if initial_velocity is not None:
			self.velocity = initial_velocity

	def set_initial_position(self, initial_position):
		self.position = initial_position

	def set_initial_velocity(self, initial_velocity):
		self.velocity = initial_velocity

	def update_position(self, new_position):
		self.position = new_position

	def update_velocity(self, new_velocity):
		self.velocity = new_velocity
