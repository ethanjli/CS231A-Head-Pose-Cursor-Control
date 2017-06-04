import numpy as np

class Chainable(object):
    """Abstract base class for nodes of chains, such as computation chains."""
    def __init__(self):
        self.source = None
        self.destination = None
        self._needs_update = False

    def register_source(self, source_parameter):
        """Registers a source Chainable with this object."""
        self.source = source_parameter

    def register_destination(self, destination_parameter):
        """Registers a destination Chainable with this object."""
        self.destination = destination_parameter

def chain(*chainables):
    """Chains Chainables from left to right, source to destination."""
    for (source, destination) in zip(chainables, chainables[1:]):
        source.register_destination(destination)
        destination.register_source(source)

class UpdatableChainable(Chainable):
    """Abstract base class for updatable elements of chains.
    This allows tracking of when stateful chainables have been modified."""
    def update(self):
        """Updates the destination based on the source."""
        pass

    def updated(self):
        """Records that there are no outstanding upstream changes remaining.
        Should be called after an update."""
        self._needs_update = False
        if self.source is not None and self.source.needs_update():
            self.source.updated()

    def needs_update(self):
        """Checks whether calling update would trigger any downstream changes.
        A change would be triggered in the evaluation of the computation chain
        if the state of the current Chainable or any upstream Chainables has changed."""
        source_needs_update = (self.source is not None
                               and self.source.needs_update())
        return self._needs_update or source_needs_update

class Parameter(UpdatableChainable):
    """Plain old parameters. Can act as sources and destinations.
    When acting as a source, it's the input to the chain.
    When acting as a destination, it's the output from the chain; then the chain
    can be executed by calling the update method."""
    def get(self):
        """Returns the value of the Parameter."""
        pass

    def set(self, value):
        """Changes the value of the Parameter."""
        pass

    def update(self):
        if self.source is not None:
            self.set(self.source.get())
        if self.destination is not None:
            self.destination.update()
        self.updated()

class ParameterPreprocessor(UpdatableChainable):
    """Abstract base class for composable functions in computation chains.
    The value from the source parameter is preprocessed, and the result
    is saved to the destination parameter.
    """
    def preprocess(self, value):
        """Returns the result of preprocessing the provided value."""
        pass

    def unpreprocess(self, value):
        """Returns the result of inverting the preprocessing of the provided value."""
        pass

    def get(self):
        """Returns the preprocessed result."""
        return self.preprocess(self.source.get())

    def update(self):
        """Causes the chain to update.
        Actually lazily delegates the work of updating to its destination."""
        if self.destination is not None:
            self.destination.update()

class ParameterOffset(ParameterPreprocessor):
    """A parameter preprocessor which adds a constant to its input to generate output."""
    def __init__(self, initial_offset):
        super(ParameterOffset, self).__init__()
        self._offset = initial_offset

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, new_offset):
        changed = new_offset != self._offset
        if isinstance(changed, np.ndarray):
            changed = np.any(changed)
        if changed:
            self._needs_update = True
        self._offset = new_offset

    def preprocess(self, value):
        return value + self.offset

    def unpreprocess(self, value):
        return value - self.offset

    def get(self):
        """Returns the preprocessed result.
        If no source was registered, takes an implicit constant source
        with value nullity."""
        if self.source is None:
            return self.offset
        return super(ParameterOffset, self).get()
