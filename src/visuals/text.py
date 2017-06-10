import vispy.scene

class Text(object):
    def __init__(self, pipeline, initial_text='', font_size=8):
        self.transformSystem = pipeline.transformSystem
        self.visual = vispy.scene.visuals.Text(
            initial_text, parent=pipeline.scene, font_size=font_size, color='red',
            anchor_x='left', anchor_y='top')

    def set_position(self, position):
        self.visual.pos = position

    def get_size(self):
        return self.visual.font_size

    def update(self):
        pass

class FramerateCounter(Text):
    def __init__(self, pipeline, framerate_counter, name='framerate', units='fps',
                 update_interval=60, draw=False):
        super(FramerateCounter, self).__init__(pipeline)
        self.framerate_counter = framerate_counter
        self.visual.draw(self.transformSystem)
        self.name = name
        self.units = units
        self.draw = draw
        self.update_counter = 0
        self.update_interval = 60

    def update(self):
        self.update_counter = (self.update_counter + 1) % self.update_interval
        if self.update_counter != 0:
            return
        framerate = self.framerate_counter.query()
        if framerate is not None:
            text = '{}: {:.2f} {}'.format(
                self.name, framerate, self.units)
            if self.draw:
                self.visual.text = text
                self.visual.draw(self.transformSystem)
            else:
                print(text)
