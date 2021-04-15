class _AuxLosses:
    def __init__(self):
        self._losses = {}
        self._loss_alphas = {}
        self._is_active = False
        self._obs = None

    def clear(self):
        self._losses.clear()
        self._loss_alphas.clear()
        self._obs = None

    def register_loss(self, name, loss, alpha=1.0):
        assert self.is_active()
        assert name not in self._losses

        self._losses[name] = loss
        self._loss_alphas[name] = alpha

    def get_loss(self, name):
        return self._losses[name]

    def reduce(self):
        assert self.is_active()
        total = 0.0

        for k in self._losses.keys():
            total = total + self._loss_alphas[k] * self._losses[k]

        return total

    def is_active(self):
        return self._is_active

    def activate(self):
        self._is_active = True

    def deactivate(self):
        self._is_active = False

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, new_obs):
        self._obs = new_obs


AuxLosses = _AuxLosses()