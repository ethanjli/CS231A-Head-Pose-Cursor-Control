import vispy.scene
import vispy.visuals

import visuals

class AxesVisual(visuals.CustomVisual, vispy.scene.visuals.XYZAxis):
    def __init__(self, *args, **kwargs):
        super(AxesVisual, self).__init__(*args, **kwargs)

    @staticmethod
    def base_transform():
        transform = vispy.visuals.transforms.AffineTransform()
        transform.scale((540, 540, 540))
        return transform
