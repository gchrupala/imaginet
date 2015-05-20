import sys
import cPickle

def deserialize(model):
    model_class = getattr(sys.modules[model['model_module']], model['model_name'])
    model['config']['layers'] = [getattr(sys.modules[layer['layer_module']], layer['layer_name'])(**layer['config']) 
                                         for layer in model['config']['layers']]
    return model_class(**model['config'])

def serialize(model):
    layer_configs = []
    for layer in model.layers:
        layer_config = layer.settings
        layer_name = layer.__class__.__name__
        layer_module = layer.__class__.__module__
        weights = [p.get_value() for p in layer.params]
        layer_config['weights'] = weights
        layer_configs.append({'layer_name': layer_name, 'layer_module': layer_module, 'config': layer_config})
    model.settings['layers'] = layer_configs
    return  {'model_name': model.__class__.__name__, 'model_module': model.__class__.__module__, 'config': model.settings}


