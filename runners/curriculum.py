import yaml
import yaml.constructor
from collections import OrderedDict


class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                                                    'expected a mapping node, but found %s' %
                                                    node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError('while constructing a mapping',
                                                        node.start_mark,
                                                        'found unacceptable key (%s)' % exc,
                                                        key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


class Task(object):

    def __init__(self, name, prop):
        self.name = name
        self.prop = prop

    def __hash__(self):
        return hash(self.name)


class Curriculum(object):

    def __init__(self, config):
        with open(config, 'r') as f:
            self.config = yaml.load(f, OrderedDictYAMLLoader)

        self._tasks = list([Task(k, v) for k, v in self.config['tasks'].items()])
        self._lesson_threshold = self.config['thresholds']['lesson']
        self._stop_threshold = self.config['thresholds']['stop']
        self._n_trials = self.config['n_trials']
        self._metric = self.config['metric']

    @property
    def tasks(self):
        return self._tasks

    @property
    def lesson_threshold(self):
        return self._lesson_threshold

    @property
    def stop_threshold(self):
        return self._stop_threshold

    @property
    def n_trials(self):
        return self._n_trials

    @property
    def metric(self):
        return self._metric
