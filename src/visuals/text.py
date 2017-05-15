import vispy.scene

class Text(object):
    def __init__(self, pipeline, initial_text='', font_size=8, color='red'):
        self.transformSystem = pipeline.transformSystem
        self.visual = vispy.scene.visuals.Text(
            initial_text, parent=pipeline.scene, font_size=font_size, color=color,
            anchor_x='left', anchor_y='top')

    def set_position(self, position):
        self.visual.pos = position

    def get_size(self):
        return self.visual.font_size

    def update(self):
        self.visual.draw(self.transformSystem)

class FramerateCounter(Text):
    def __init__(self, pipeline, framerate_counter, name='framerate', units='fps'):
        super(FramerateCounter, self).__init__(pipeline)
        self.framerate_counter = framerate_counter
        self.visual.draw(self.transformSystem)
        self.name = name
        self.units = units

    def update(self):
        framerate = self.framerate_counter.query()
        if framerate is not None:
            self.visual.text = '{}: {:.2f} {}'.format(
                self.name, framerate, self.units)
            self.visual.draw(self.transformSystem)
