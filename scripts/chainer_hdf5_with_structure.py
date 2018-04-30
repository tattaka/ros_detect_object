#!/usr/bin/env python
# -*- coding: utf-8 -*-
import chainer
import numpy
import six
from YOLOv2.yolov2 import *
# based on
# https://stackoverflow.com/questions/452969/does-python-have-an-equivalent-to-java-class-forname#30042585
def import_class_from_string(path):
    from importlib import import_module
    module_path, _, class_name = path.rpartition('.')
    if module_path == '':
        klass = globals()[class_name]
    else:
        mod = import_module(module_path)
        klass = getattr(mod, class_name)
    return klass

def _convert_construction_info_to_string(obj):
    """
    Generate a string including information required to initialize an object
    in the same way of that of obj.

    Parameters
    ----------
    obj : object
        An object.
        It must have the attribute `_constructor_args` that represents
        the arguments of the constructor that has been used for initializing
        the object.

    Returns
    ----------
    encoded_construction_info : string
        It represents the module, class name and constructor arguments of obj.
    """
    if not hasattr(obj, '_constructor_args'):
        raise AttributeError('obj has no attribute _constructor_args.')
    import StringIO
    output = StringIO.StringIO()
    info = {}
    info['_module_name'] = obj.__class__.__module__
    info['_class_name'] = obj.__class__.__name__
    encoded_constructor_args = {}
    for k, v in obj._constructor_args.items():
        if isinstance(v, chainer.Link):
            encoded_constructor_args[k] \
                = _convert_construction_info_to_string(v)
        elif isinstance(v, six.string_types):
            encoded_constructor_args[k] = 'STR' + v
        else:
            encoded_constructor_args[k] = v
    info['_encoded_constructor_args'] = encoded_constructor_args
    numpy.save(output, arr=info)
    encoded_construction_info = 'OBJ' + output.getvalue()
    output.close()
    return encoded_construction_info

def _restore_obj_from_construction_info_string(str, class_name_replacement_list = None):
    """
    Restore an object from a construction info string.

    Parameters
    ----------
    str : string
        It represents the module, class name and constructor arguments
        of an object.

    class_name_replacement_list : list
        A list of a tuple of pattern and replacement.
        If a class name with a module matches a pattern in the list,
        the string is replaced by the replacement.
        The patterns are compared in the order of the list and only
        the first replacement will be applied.

    Returns
    -------
    obj : object
        It is initialized according to the construction info.
    """
    if str[0:3] == 'STR':
        obj = str[3:]
    elif str[0:3] == 'OBJ':
        import StringIO
        inp = StringIO.StringIO(str[3:])
        info = numpy.load(inp).item()
        inp.close()
        constructor_args = {}
        # Decode the encoded constructor arguments.
        for k, v in info['_encoded_constructor_args'].items():
            if isinstance(v, six.string_types):
                constructor_args[k] \
                    = _restore_obj_from_construction_info_string(
                        v, class_name_replacement_list)
            else:
                constructor_args[k] = v
        module_name = info['_module_name']
        class_name = info['_class_name']
        full_class_name = module_name + '.' + class_name
        original_full_class_name = full_class_name
        if not class_name_replacement_list is None:
            import re
            for pat, repl in class_name_replacement_list:
                if re.match(pat, full_class_name):
                    full_class_name = re.sub(pat, repl, full_class_name)
                    break
        class_object = import_class_from_string(full_class_name)
        obj = make_serializable_object(class_object, constructor_args)
    else:
        raise ValueError()
    return obj

def make_serializable_object(class_object, constructor_args, template_args=None):
    """
    Generate an object and attach the argument of the constructor to
    to the object.

    Parameters
    ----------
    class_object : class object
    constructor_args : dict
        Arguments used for generating an object of the given class.
    template_args : dict
        Arguments used for generating a template of an object of the
        given class in advance of loading trained weights.

    Returns
    -------
    obj : object
        It has an additional attribute `_constructor_args` that refers
        to template_args or constructor_args.
    """
    obj = class_object(**constructor_args)
    obj._constructor_args = template_args or constructor_args
    return obj

def load_hdf5_with_structure(file):
    """
    load model from a '.model' file.
    """
    n_classes = 80
    n_boxes = 5
    anchors = [[0.738768, 0.874946], [2.42204, 2.65704], [4.30971, 7.04493], [10.246, 4.59428], [12.6868, 11.8741]]

    yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
    chainer.serializers.load_hdf5(file, yolov2)
    model = YOLOv2Predictor(yolov2)
    model.init_anchor(anchors)
    #model.predictor.train = False
    model.predictor.finetune = False
    return model
