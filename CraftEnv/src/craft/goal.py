class Goal:
    def __init__(self, blackboard):
        self._blackboard = blackboard
        self.reset()

    def reset(self):
        self.template = self._blackboard.template
        self.x = self.template["goal"][0]["x"]
        self.y = self.template["goal"][0]["y"]
        self.z = self.template["goal"][0]["z"]
