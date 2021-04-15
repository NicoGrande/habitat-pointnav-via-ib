import numpy as np
from habitat_sim.utils.common import quat_rotate_vector, quat_from_angle_axis
from habitat_sim import geo


class SE3_Noise:
    def __init__(self, rot=None, trans=None):
        self.rot = rot if rot is not None else quat_from_angle_axis(0.0, geo.UP)
        self.trans = trans if trans is not None else np.array([0.0, 0.0, 0.0])

    def __mul__(self, other):
        if isinstance(other, SE3_Noise):
            return SE3_Noise(
                self.rot * other.rot,
                self.trans + quat_rotate_vector(self.rot, other.trans),
            )
        else:
            return quat_rotate_vector(self.rot, other) + self.trans

    def inverse(self):
        return SE3_Noise(
            self.rot.inverse(), quat_rotate_vector(self.rot.inverse(), -self.trans)
        )