''' Properties are objects that can be assigned as class attributes on Bokeh
models, to provide automatic serialization, validation, and documentation.

There are many property types defined in the module, for example ``Int`` to
represent integral values, ``Seq`` to represent sequences (e.g. lists or
tuples, etc.). Properties can also be combined: ``Seq(Float)`` represents
a sequence of floating point values.

For example, the following defines a model that has integer, string, and
list[float] properties:

.. code-block:: python

    class SomeModel(Model):
        foo = Int
        bar = String(default="something")
        baz = List(Float, help="docs for baz prop")

As seen, properties can be declared as just the property type, e.g.
``foo = Int``, in which case the properties are automatically instantiated
on new Model objects. Or the property can be instantiated on the class,
and configured with default values and help strings.

The properties of this class can be initialized by specifying keyword
arguments to the initializer:

.. code-block:: python

    m = SomeModel(foo=10, bar="a str", baz=[1,2,3,4])

But also by setting the attributes on an instance:

.. code-block:: python

    m.foo = 20

Attempts to set a property to a value of the wrong type will
result in a ``ValueError`` exception:

.. code-block:: python

    >>> m.foo = 2.3
    Traceback (most recent call last):

      << traceback omitted >>

    ValueError: expected a value of type Integral, got 2.3 of type float

Models with properties know how to serialize themselves, to be understood
by BokehJS. Additionally, any help strings provided on properties can be
easily and automatically extracted with the Sphinx extensions in the
:ref:`bokeh.sphinxext` module.

.. |Color| replace:: :func:`~bokeh.core.properties.Color`
.. |DataSpec| replace:: :func:`~bokeh.core.properties.DataSpec`
.. |field| replace:: :func:`~bokeh.core.properties.field`
.. |value| replace:: :func:`~bokeh.core.properties.value`


'''
from __future__ import absolute_import, print_function

import logging
logger = logging.getLogger(__name__)

import collections
from copy import copy
import datetime
import dateutil.parser
from importlib import import_module
import numbers
import re

from six import string_types, iteritems

from ..colors import RGB
from ..util.dependencies import import_optional
from ..util.deprecation import deprecated
from ..util.serialization import transform_column_source_data, decode_base64_dict
from ..util.string import nice_join
from .property.override import Override ; Override # TODO
from .property.bases import (
    ContainerProperty, ParameterizedProperty, Property,
    PropertyFactory,  PrimitiveProperty, DeserializationError, MODEL_LINK, PROP_LINK
    )
from .property.descriptors import BasicPropertyDescriptor, DataSpecPropertyDescriptor, UnitsSpecPropertyDescriptor
from . import enums

pd = import_optional('pandas')

def field(name):
    ''' Convenience function to explicitly return a "field" specification for
    a Bokeh :class:`~bokeh.core.properties.DataSpec` property.

    Args:
        name (str) : name of a data source field to reference for a
            ``DataSpec`` property.

    Returns:
        dict : ``{"field": name}``

    .. note::
        This function is included for completeness. String values for
        property specifications are by default interpreted as field names.

    '''
    return dict(field=name)

def value(val):
    ''' Convenience function to explicitly return a "value" specification for
    a Bokeh :class:`~bokeh.core.properties.DataSpec` property.

    Args:
        val (any) : a fixed value to specify for a ``DataSpec`` property.

    Returns:
        dict : ``{"value": name}``

    ..note::
        String values for property specifications are by default interpreted
        as field names. This function is especially useful when you want to
        specify a fixed value with text properties.

    Example:

    .. code-block:: python

        # The following will take text values to render from a data source
        # column "text_column", but use a fixed value "12pt" for font size
        p.text("x", "y", text="text_column",
               text_font_size=value("12pt"), source=source)

    '''
    return dict(value=val)

def abstract(cls):
    from .has_props import HasProps
    ''' A phony decorator to mark abstract base classes. '''
    if not issubclass(cls, HasProps):
        raise TypeError("%s is not a subclass of HasProps" % cls.__name__)

    return cls

bokeh_bool_types = (bool,)
try:
    import numpy as np
    bokeh_bool_types += (np.bool8,)
except ImportError:
    pass

bokeh_integer_types = (numbers.Integral,)


class Include(PropertyFactory):
    ''' Include other properties from mixin Models, with a given prefix. '''

    def __init__(self, delegate, help="", use_prefix=True):
        from .has_props import HasProps
        if not (isinstance(delegate, type) and issubclass(delegate, HasProps)):
            raise ValueError("expected a subclass of HasProps, got %r" % delegate)

        self.delegate = delegate
        self.help = help
        self.use_prefix = use_prefix

    def make_descriptors(self, base_name):
        descriptors = []
        delegate = self.delegate
        if self.use_prefix:
            prefix = re.sub("_props$", "", base_name) + "_"
        else:
            prefix = ""

        # it would be better if we kept the original generators from
        # the delegate and built our Include props from those, perhaps.
        for subpropname in delegate.properties(with_bases=False):
            fullpropname = prefix + subpropname
            subprop_descriptor = delegate.lookup(subpropname)
            if isinstance(subprop_descriptor, BasicPropertyDescriptor):
                prop = copy(subprop_descriptor.property)
                if "%s" in self.help:
                    doc = self.help % subpropname.replace('_', ' ')
                else:
                    doc = self.help
                prop.__doc__ = doc
                descriptors += prop.make_descriptors(fullpropname)

        return descriptors

class Bool(PrimitiveProperty):
    ''' Boolean type property. '''
    _underlying_type = bokeh_bool_types

class Int(PrimitiveProperty):
    ''' Signed integer type property. '''
    _underlying_type = bokeh_integer_types

class Float(PrimitiveProperty):
    ''' Floating point type property. '''
    _underlying_type = (numbers.Real,)

class Complex(PrimitiveProperty):
    ''' Complex floating point type property. '''
    _underlying_type = (numbers.Complex,)

class String(PrimitiveProperty):
    ''' String type property. '''
    _underlying_type = string_types

class Regex(String):
    ''' Regex type property validates that text values match the
    given regular expression.
    '''
    def __init__(self, regex, default=None, help=None):
        self.regex = re.compile(regex)
        super(Regex, self).__init__(default=default, help=help)

    def validate(self, value):
        super(Regex, self).validate(value)

        if not (value is None or self.regex.match(value) is not None):
            raise ValueError("expected a string matching %r pattern, got %r" % (self.regex.pattern, value))

    def __str__(self):
        return "%s(%r)" % (self.__class__.__name__, self.regex.pattern)

class JSON(String):
    ''' JSON type property validates that text values are valid JSON.

    ..  note::
        The string is transmitted and received by BokehJS as a *string*
        containing JSON content. i.e., you must use ``JSON.parse`` to unpack
        the value into a JavaScript hash.

    '''
    def validate(self, value):
        super(JSON, self).validate(value)

        if value is None: return

        try:
            import json
            json.loads(value)
        except ValueError:
            raise ValueError("expected JSON text, got %r" % value)

class Seq(ContainerProperty):
    ''' An ordered sequence of values (list, tuple, (nd)array). '''

    @classmethod
    def _is_seq(cls, value):
        return ((isinstance(value, collections.Sequence) or cls._is_seq_like(value)) and
                not isinstance(value, string_types))

    @classmethod
    def _is_seq_like(cls, value):
        return (isinstance(value, (collections.Container, collections.Sized, collections.Iterable))
                and hasattr(value, "__getitem__") # NOTE: this is what makes it disallow set type
                and not isinstance(value, collections.Mapping))

    def _new_instance(self, value):
        return value

    def __init__(self, item_type, default=None, help=None):
        self.item_type = self._validate_type_param(item_type)
        super(Seq, self).__init__(default=default, help=help)

    @property
    def type_params(self):
        return [self.item_type]

    def validate(self, value):
        super(Seq, self).validate(value)

        if value is not None:
            if not (self._is_seq(value) and all(self.item_type.is_valid(item) for item in value)):
                if self._is_seq(value):
                    invalid = []
                    for item in value:
                        if not self.item_type.is_valid(item):
                            invalid.append(item)
                    raise ValueError("expected an element of %s, got seq with invalid items %r" % (self, invalid))
                else:
                    raise ValueError("expected an element of %s, got %r" % (self, value))

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.item_type)

    def _sphinx_type(self):
        return PROP_LINK % self.__class__.__name__ + "( %s )" % self.item_type._sphinx_type()

    def from_json(self, json, models=None):
        if json is None:
            return None
        elif isinstance(json, list):
            return self._new_instance([ self.item_type.from_json(item, models) for item in json ])
        else:
            raise DeserializationError("%s expected a list or None, got %s" % (self, json))

class List(Seq):
    ''' Python list type property.

    '''

    def __init__(self, item_type, default=[], help=None):
        # todo: refactor to not use mutable objects as default values.
        # Left in place for now because we want to allow None to express
        # optional values. Also in Dict.
        super(List, self).__init__(item_type, default=default, help=help)

    @classmethod
    def _is_seq(self, value):
        return isinstance(value, list)

class Array(Seq):
    ''' NumPy array type property.

    '''

    @classmethod
    def _is_seq(self, value):
        import numpy as np
        return isinstance(value, np.ndarray)

    def _new_instance(self, value):
        import numpy as np
        return np.array(value)


class Dict(ContainerProperty):
    ''' Python dict type property.

    If a default value is passed in, then a shallow copy of it will be
    used for each new use of this property.

    '''

    def __init__(self, keys_type, values_type, default={}, help=None):
        self.keys_type = self._validate_type_param(keys_type)
        self.values_type = self._validate_type_param(values_type)
        super(Dict, self).__init__(default=default, help=help)

    @property
    def type_params(self):
        return [self.keys_type, self.values_type]

    def validate(self, value):
        super(Dict, self).validate(value)

        if value is not None:
            if not (isinstance(value, dict) and \
                    all(self.keys_type.is_valid(key) and self.values_type.is_valid(val) for key, val in iteritems(value))):
                raise ValueError("expected an element of %s, got %r" % (self, value))

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.keys_type, self.values_type)

    def _sphinx_type(self):
        return PROP_LINK % self.__class__.__name__ + "( %s, %s )" % (self.keys_type._sphinx_type(), self.values_type._sphinx_type())

    def from_json(self, json, models=None):
        if json is None:
            return None
        elif isinstance(json, dict):
            return { self.keys_type.from_json(key, models): self.values_type.from_json(value, models) for key, value in iteritems(json) }
        else:
            raise DeserializationError("%s expected a dict or None, got %s" % (self, json))

class ColumnData(Dict):
    '''Property holding column data in form of a dict. Also applies
    encoding and decoding to the data.
    '''

    def from_json(self, json, models=None):
        ''' Decodes column source data encoded as lists or base64 strings.
        '''
        if json is None:
            return None
        elif not isinstance(json, dict):
            raise DeserializationError("%s expected a dict or None, got %s" % (self, json))
        new_data = {}
        for key, value in json.items():
            key = self.keys_type.from_json(key, models)
            if isinstance(value, dict) and '__ndarray__' in value:
                new_data[key] = decode_base64_dict(value)
            elif isinstance(value, list) and any(isinstance(el, dict) and '__ndarray__' in el for el in value):
                new_list = []
                for el in value:
                    if isinstance(el, dict) and '__ndarray__' in el:
                        el = decode_base64_dict(el)
                    elif isinstance(el, list):
                        el = self.values_type.from_json(el)
                    new_list.append(el)
                new_data[key] = new_list
            else:
                new_data[key] = self.values_type.from_json(value, models)
        return new_data


    def serialize_value(self, value):
        return transform_column_source_data(value)

class Tuple(ContainerProperty):
    ''' Tuple type property. '''
    def __init__(self, tp1, tp2, *type_params, **kwargs):
        self._type_params = list(map(self._validate_type_param, (tp1, tp2) + type_params))
        super(Tuple, self).__init__(default=kwargs.get("default"), help=kwargs.get("help"))

    @property
    def type_params(self):
        return self._type_params

    def validate(self, value):
        super(Tuple, self).validate(value)

        if value is not None:
            if not (isinstance(value, (tuple, list)) and len(self.type_params) == len(value) and \
                    all(type_param.is_valid(item) for type_param, item in zip(self.type_params, value))):
                raise ValueError("expected an element of %s, got %r" % (self, value))

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join(map(str, self.type_params)))

    def _sphinx_type(self):
        return PROP_LINK % self.__class__.__name__ + "( %s )" % ", ".join(x._sphinx_type() for x in self.type_params)

    def from_json(self, json, models=None):
        if json is None:
            return None
        elif isinstance(json, list):
            return tuple(type_param.from_json(item, models) for type_param, item in zip(self.type_params, json))
        else:
            raise DeserializationError("%s expected a list or None, got %s" % (self, json))

class Instance(Property):
    ''' Instance type property, for references to other Models in the object
    graph.

    '''
    def __init__(self, instance_type, default=None, help=None):
        if not isinstance(instance_type, (type,) + string_types):
            raise ValueError("expected a type or string, got %s" % instance_type)

        from .has_props import HasProps
        if isinstance(instance_type, type) and not issubclass(instance_type, HasProps):
            raise ValueError("expected a subclass of HasProps, got %s" % instance_type)

        self._instance_type = instance_type

        super(Instance, self).__init__(default=default, help=help)

    @property
    def instance_type(self):
        if isinstance(self._instance_type, str):
            module, name = self._instance_type.rsplit(".", 1)
            self._instance_type = getattr(import_module(module, "bokeh"), name)

        return self._instance_type

    def _has_stable_default(self):
        # because the instance value is mutable
        return False

    @property
    def has_ref(self):
        return True

    def validate(self, value):
        super(Instance, self).validate(value)

        if value is not None:
            if not isinstance(value, self.instance_type):
                raise ValueError("expected an instance of type %s, got %s of type %s" %
                    (self.instance_type.__name__, value, type(value).__name__))

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.instance_type.__name__)

    def _sphinx_type(self):
        fullname = "%s.%s" % (self.instance_type.__module__, self.instance_type.__name__)
        return PROP_LINK % self.__class__.__name__ + "( %s )" % MODEL_LINK % fullname

    def from_json(self, json, models=None):
        if json is None:
            return None
        elif isinstance(json, dict):
            from ..model import Model
            if issubclass(self.instance_type, Model):
                if models is None:
                    raise DeserializationError("%s can't deserialize without models" % self)
                else:
                    model = models.get(json["id"])

                    if model is not None:
                        return model
                    else:
                        raise DeserializationError("%s failed to deserialize reference to %s" % (self, json))
            else:
                attrs = {}

                for name, value in iteritems(json):
                    prop_descriptor = self.instance_type.lookup(name).property
                    attrs[name] = prop_descriptor.from_json(value, models)

                # XXX: this doesn't work when Instance(Superclass) := Subclass()
                # Serialization dict must carry type information to resolve this.
                return self.instance_type(**attrs)
        else:
            raise DeserializationError("%s expected a dict or None, got %s" % (self, json))

class This(Property):
    ''' A reference to an instance of the class being defined. '''
    pass

# Fake types, ABCs
class Any(Property):
    ''' Any type property accepts any values. '''
    pass

class Function(Property):
    ''' Function type property. '''
    pass

class Event(Property):
    ''' Event type property. '''
    pass

class Interval(ParameterizedProperty):
    ''' Range type property ensures values are contained inside a given interval. '''
    def __init__(self, interval_type, start, end, default=None, help=None):
        self.interval_type = self._validate_type_param(interval_type)
        # Make up a property name for validation purposes
        self.interval_type.validate(start)
        self.interval_type.validate(end)
        self.start = start
        self.end = end
        super(Interval, self).__init__(default=default, help=help)

    @property
    def type_params(self):
        return [self.interval_type]

    def validate(self, value):
        super(Interval, self).validate(value)

        if not (value is None or self.interval_type.is_valid(value) and value >= self.start and value <= self.end):
            raise ValueError("expected a value of type %s in range [%s, %s], got %r" % (self.interval_type, self.start, self.end, value))

    def __str__(self):
        return "%s(%s, %r, %r)" % (self.__class__.__name__, self.interval_type, self.start, self.end)

class Byte(Interval):
    ''' Byte type property. '''
    def __init__(self, default=0, help=None):
        super(Byte, self).__init__(Int, 0, 255, default=default, help=help)

class Either(ParameterizedProperty):
    ''' Takes a list of valid properties and validates against them in succession. '''

    def __init__(self, tp1, tp2, *type_params, **kwargs):
        self._type_params = list(map(self._validate_type_param, (tp1, tp2) + type_params))
        help = kwargs.get("help")
        def choose_default():
            return self._type_params[0]._raw_default()
        default = kwargs.get("default", choose_default)
        super(Either, self).__init__(default=default, help=help)

    @property
    def type_params(self):
        return self._type_params

    def validate(self, value):
        super(Either, self).validate(value)

        if not (value is None or any(param.is_valid(value) for param in self.type_params)):
            raise ValueError("expected an element of either %s, got %r" % (nice_join(self.type_params), value))

    def transform(self, value):
        for param in self.type_params:
            try:
                return param.transform(value)
            except ValueError:
                pass

        raise ValueError("Could not transform %r" % value)

    def from_json(self, json, models=None):
        for tp in self.type_params:
            try:
                return tp.from_json(json, models)
            except DeserializationError:
                pass
        else:
            raise DeserializationError("%s couldn't deserialize %s" % (self, json))

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join(map(str, self.type_params)))

    def _sphinx_type(self):
        return PROP_LINK % self.__class__.__name__ + "( %s )" % ", ".join(x._sphinx_type() for x in self.type_params)

    def __or__(self, other):
        return self.__class__(*(self.type_params + [other]), default=self._default, help=self.help)

class Enum(String):
    ''' An Enum with a list of allowed values. The first value in the list is
    the default value, unless a default is provided with the "default" keyword
    argument.
    '''
    def __init__(self, enum, *values, **kwargs):
        if not (not values and isinstance(enum, enums.Enumeration)):
            enum = enums.enumeration(enum, *values)

        self._enum = enum

        default = kwargs.get("default", enum._default)
        help = kwargs.get("help")

        super(Enum, self).__init__(default=default, help=help)

    @property
    def allowed_values(self):
        return self._enum._values

    def validate(self, value):
        super(Enum, self).validate(value)

        if not (value is None or value in self._enum):
            raise ValueError("invalid value: %r; allowed values are %s" % (value, nice_join(self.allowed_values)))

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join(map(repr, self.allowed_values)))

    def _sphinx_type(self):
        # try to return a link to a proper enum in bokeh.core.enums if possible
        if self._enum in enums.__dict__.values():
            for name, obj in enums.__dict__.items():
                if self._enum is obj:
                    val = MODEL_LINK % "%s.%s" % (self._enum.__module__, name)
        else:
            val = str(self._enum)
        return PROP_LINK % self.__class__.__name__ + "( %s )" % val

class Auto(Enum):
    ''' Accepts the string "auto".

    Useful for properties that can be configured to behave "automatically".

    '''
    def __init__(self):
        super(Auto, self).__init__("auto")

    def __str__(self):
        return self.__class__.__name__

    def _sphinx_type(self):
        return PROP_LINK % self.__class__.__name__

# Properties useful for defining visual attributes
class Color(Either):
    ''' Accepts color definition in a variety of ways, and produces an
    appropriate serialization of its value for whatever backend.

    For colors, because we support named colors and hex values prefaced
    with a "#", when we are handed a string value, there is a little
    interpretation: if the value is one of the 147 SVG named colors or
    it starts with a "#", then it is interpreted as a value.

    If a 3-tuple is provided, then it is treated as an RGB (0..255).
    If a 4-tuple is provided, then it is treated as an RGBa (0..255), with
    alpha as a float between 0 and 1.  (This follows the HTML5 Canvas API.)
    '''

    def __init__(self, default=None, help=None):
        types = (Enum(enums.NamedColor),
                 Regex("^#[0-9a-fA-F]{6}$"),
                 Tuple(Byte, Byte, Byte),
                 Tuple(Byte, Byte, Byte, Percent))
        super(Color, self).__init__(*types, default=default, help=help)

    def transform(self, value):
        if isinstance(value, tuple):
            value = RGB(*value).to_css()
        return value

    def __str__(self):
        return self.__class__.__name__

    def _sphinx_type(self):
        return PROP_LINK % self.__class__.__name__


class MinMaxBounds(Either):
    ''' Accepts min and max bounds for use with Ranges.

    Bounds are provided as a tuple of ``(min, max)`` so regardless of whether your range is
    increasing or decreasing, the first item should be the minimum value of the range and the
    second item should be the maximum. Setting min > max will result in a ``ValueError``.

    Setting bounds to None will allow your plot to pan/zoom as far as you want. If you only
    want to constrain one end of the plot, you can set min or max to
    ``None`` e.g. ``DataRange1d(bounds=(None, 12))`` '''

    def __init__(self, accept_datetime=False, default='auto', help=None):
        if accept_datetime:
            types = (
                Auto,
                Tuple(Float, Float),
                Tuple(Datetime, Datetime),
            )
        else:
            types = (
                Auto,
                Tuple(Float, Float),
            )
        super(MinMaxBounds, self).__init__(*types, default=default, help=help)

    def validate(self, value):
        super(MinMaxBounds, self).validate(value)

        if value is None:
            pass

        elif value[0] is None or value[1] is None:
            pass

        elif value[0] >= value[1]:
            raise ValueError('Invalid bounds: maximum smaller than minimum. Correct usage: bounds=(min, max)')

        return True

    def _sphinx_type(self):
        return PROP_LINK % self.__class__.__name__


class Align(Property):
    pass


class DashPattern(Either):
    ''' Dash type property.

    Express patterns that describe line dashes.  ``DashPattern`` values
    can be specified in a variety of ways:

    * An enum: "solid", "dashed", "dotted", "dotdash", "dashdot"
    * a tuple or list of integers in the `HTML5 Canvas dash specification style`_.
      Note that if the list of integers has an odd number of elements, then
      it is duplicated, and that duplicated list becomes the new dash list.

    To indicate that dashing is turned off (solid lines), specify the empty
    list [].

    .. _HTML5 Canvas dash specification style: http://www.w3.org/html/wg/drafts/2dcontext/html5_canvas/#dash-list

    '''

    _dash_patterns = {
        "solid": [],
        "dashed": [6],
        "dotted": [2,4],
        "dotdash": [2,4,6,4],
        "dashdot": [6,4,2,4],
    }

    def __init__(self, default=[], help=None):
        types = Enum(enums.DashPattern), Regex(r"^(\d+(\s+\d+)*)?$"), Seq(Int)
        super(DashPattern, self).__init__(*types, default=default, help=help)

    def transform(self, value):
        value = super(DashPattern, self).transform(value)

        if isinstance(value, string_types):
            try:
                return self._dash_patterns[value]
            except KeyError:
                return [int(x) for x in  value.split()]
        else:
            return value

    def __str__(self):
        return self.__class__.__name__

    def _sphinx_type(self):
        return PROP_LINK % self.__class__.__name__

class Size(Float):
    ''' Size type property.

    .. note::
        ``Size`` is equivalent to an unsigned int.

    '''
    def validate(self, value):
        super(Size, self).validate(value)

        if not (value is None or 0.0 <= value):
            raise ValueError("expected a non-negative number, got %r" % value)

class Percent(Float):
    ''' Percentage type property.

    Percents are useful for specifying alphas and coverage and extents; more
    semantically meaningful than Float(0..1).

    '''
    def validate(self, value):
        super(Percent, self).validate(value)

        if not (value is None or 0.0 <= value <= 1.0):
            raise ValueError("expected a value in range [0, 1], got %r" % value)

class Angle(Float):
    ''' Angle type property. '''
    pass

class Date(Property):
    ''' Date (not datetime) type property.

    '''
    def __init__(self, default=datetime.date.today(), help=None):
        super(Date, self).__init__(default=default, help=help)

    def validate(self, value):
        super(Date, self).validate(value)

        if not (value is None or isinstance(value, (datetime.date,) + string_types + (float,) + bokeh_integer_types)):
            raise ValueError("expected a date, string or timestamp, got %r" % value)

    def transform(self, value):
        value = super(Date, self).transform(value)

        if isinstance(value, (float,) + bokeh_integer_types):
            try:
                value = datetime.date.fromtimestamp(value)
            except ValueError:
                value = datetime.date.fromtimestamp(value/1000)
        elif isinstance(value, string_types):
            value = dateutil.parser.parse(value).date()

        return value

class Datetime(Property):
    ''' Datetime type property.

    '''

    def __init__(self, default=datetime.date.today(), help=None):
        super(Datetime, self).__init__(default=default, help=help)

    def validate(self, value):
        super(Datetime, self).validate(value)

        datetime_types = (datetime.datetime, datetime.date)
        try:
            import numpy as np
            datetime_types += (np.datetime64,)
        except (ImportError, AttributeError) as e:
            if e.args == ("'module' object has no attribute 'datetime64'",):
                import sys
                if 'PyPy' in sys.version:
                    pass
                else:
                    raise e
            else:
                pass

        if (isinstance(value, datetime_types)):
            return

        if pd and isinstance(value, (pd.Timestamp)):
            return

        raise ValueError("Expected a datetime instance, got %r" % value)

    def transform(self, value):
        value = super(Datetime, self).transform(value)
        return value
        # Handled by serialization in protocol.py for now

class TimeDelta(Property):
    ''' TimeDelta type property.

    '''

    def __init__(self, default=datetime.timedelta(), help=None):
        super(TimeDelta, self).__init__(default=default, help=help)

    def validate(self, value):
        super(TimeDelta, self).validate(value)

        timedelta_types = (datetime.timedelta,)
        try:
            import numpy as np
            timedelta_types += (np.timedelta64,)
        except (ImportError, AttributeError) as e:
            if e.args == ("'module' object has no attribute 'timedelta64'",):
                import sys
                if 'PyPy' in sys.version:
                    pass
                else:
                    raise e
            else:
                pass

        if (isinstance(value, timedelta_types)):
            return

        if pd and isinstance(value, (pd.Timedelta)):
            return

        raise ValueError("Expected a timedelta instance, got %r" % value)

    def transform(self, value):
        value = super(TimeDelta, self).transform(value)
        return value
        # Handled by serialization in protocol.py for now

class TitleProp(Either):
    ''' Accepts a title for a plot (possibly transforming a plain string).

    .. note::
        This property exists only to support a deprecation, and will be removed
        in the future once the deprecation is completed.

    '''
    def __init__(self, default=None, help=None):
        types = (Instance('bokeh.models.annotations.Title'), String)
        super(TitleProp, self).__init__(*types, default=default, help=help)

    def _sphinx_type(self):
        return PROP_LINK % self.__class__.__name__

    def transform(self, value):
        if isinstance(value, str):
            from bokeh.models.annotations import Title
            deprecated('''Setting Plot property 'title' using a string was deprecated in 0.12.0,
            and will be removed. The title is now an object on Plot (which holds all of it's
            styling properties). Please use Plot.title.text instead.

            SERVER USERS: If you were using plot.title to have the server update the plot title
            in a callback, you MUST update to plot.title.text as the title object cannot currently
            be replaced after initialization.
            ''')
            value = Title(text=value)
        return value

class RelativeDelta(Dict):
    ''' RelativeDelta type property for time deltas.

    '''

    def __init__(self, default={}, help=None):
        keys = Enum("years", "months", "days", "hours", "minutes", "seconds", "microseconds")
        values = Int
        super(RelativeDelta, self).__init__(keys, values, default=default, help=help)

    def __str__(self):
        return self.__class__.__name__

class DataSpec(Either):
    ''' Base class for properties that can represent either a fixed value,
    or a reference to a column in a data source.

    Many Bokeh models have properties that a user might want to set either
    to a single fixed value, or to have the property take values from some
    column in a data source. As a concrete example consider a glyph with
    an ``x`` property for location. We might want to set all the glyphs
    that get drawn to have the same location, say ``x=10``. It would be
    convenient to  just be able to write:

    .. code-block:: python

        glyph.x = 10

    Alternatively, maybe the each glyph that gets drawn should have a
    different location, according to the "pressure" column of a data
    source. In this case we would like to be able to write:

    .. code-block:: python

        glyph.x = "pressure"

    Bokeh ``DataSpec`` properties (and subclasses) afford this ease of
    and consistency of expression. Ultimately, all ``DataSpec`` properties
    resolve to dictionary values, with either a ``"value"`` key, or a
    ``"field"`` key, depending on how it is set.

    For instance:

    .. code-block:: python

        glyph.x = 10          # => { 'value': 10 }

        glyph.x = "pressure"  # => { 'field': 'pressure' }

    When these underlying dictionary dictionary values are received in
    the browser, BokehJS knows how to interpret them and take the correct,
    expected action (i.e., draw the glyph at ``x=10``, or draw the glyph
    with ``x`` coordinates from the "pressure" column). In this way, both
    use-cases may be expressed easily in python, without having to handle
    anything differently, from the user perspective.

    It is worth noting that ``DataSpec`` properties can also be set directly
    with properly formed dictionary values:

    .. code-block:: python

        glyph.x = { 'value': 10 }         # same as glyph.x = 10

        glyph.x = { 'field': 'pressure' } # same as glyph.x = "pressure"

    Setting the property directly as a dict can be useful in certain
    situations. For instance some ``DataSpec`` subclasses also add a
    ``"units"`` key to the dictionary. This key is often set automatically,
    but the dictionary format provides a direct mechanism to override as
    necessary. Additionally, ``DataSpec`` can have a ``"transform"`` key,
    that specifies a client-side transform that should be applied to any
    fixed or field values before they are uses. As an example, you might want
    to apply a ``Jitter`` transform to the ``x`` values:

    .. code-block:: python

        glyph.x = { 'value': 10, 'transform': Jitter(width=0.4) }

    Note that ``DataSpec`` is not normally useful on its own. Typically,
    a model will define properties using one of the sublclasses such
    as :class:`~bokeh.core.properties.NumberSpec` or
    :class:`~bokeh.core.properties.ColorSpec`. For example, a Bokeh
    model with ``x``, ``y`` and ``color`` properties that can handle
    fixed values or columns automatically might look like:

    .. code-block:: python

        class SomeModel(Model):

            x = NumberSpec(default=0, help="docs for x")

            y = NumberSpec(default=0, help="docs for y")

            color = ColorSpec(help="docs for color") # defaults to None

    '''
    def __init__(self, typ, default, help=None):
        super(DataSpec, self).__init__(
            String,
            Dict(
                String,
                Either(
                    String,
                    Instance('bokeh.models.transforms.Transform'),
                    Instance('bokeh.models.mappers.ColorMapper'),
                    typ)),
            typ,
            default=default,
            help=help
        )
        self._type = self._validate_type_param(typ)

    # TODO (bev) add stricter validation on keys

    def make_descriptors(self, base_name):
        return [ DataSpecPropertyDescriptor(property=self, name=base_name) ]

    def to_serializable(self, obj, name, val):
        # Check for None value; this means "the whole thing is
        # unset," not "the value is None."
        if val is None:
            return None

        # Check for spec type value
        try:
            self._type.validate(val)
            return dict(value=val)
        except ValueError:
            pass

        # Check for data source field name
        if isinstance(val, string_types):
            return dict(field=val)

        # Must be dict, return as-is
        return val

    def _sphinx_type(self):
        return PROP_LINK % self.__class__.__name__

class NumberSpec(DataSpec):
    ''' A |DataSpec| property that can be set to a fixed value that is a
    number, or to a data source column name referring to a column of
    numeric data.

    .. code-block:: python

        m.location = 10.3  # value

        m.location = "foo" # field

    '''
    def __init__(self, default=None, help=None):
        super(NumberSpec, self).__init__(Float, default=default, help=help)

class StringSpec(DataSpec):
    ''' A |DataSpec| property that can be set to a fixed value that is a
    string, or to a data source column name referring to a column of string
    data.

    Because acceptable fixed values and field names are both strings, it can
    be necessary explicitly to disambiguate these possibilities. By default,
    string values are interpreted as fields, but the |value| function can be
    used to specify that a string should interpreted as a value:

    .. code-block:: python

        m.title = value("foo") # value

        m.title = "foo"        # field

    '''
    def __init__(self, default, help=None):
        super(StringSpec, self).__init__(List(String), default=default, help=help)

    def prepare_value(self, cls, name, value):
        if isinstance(value, list):
            if len(value) != 1:
                raise TypeError("StringSpec convenience list values must have length 1")
            value = dict(value=value[0])
        return super(StringSpec, self).prepare_value(cls, name, value)

class FontSizeSpec(DataSpec):
    ''' A |DataSpec| property that can be set to a fixed value that is a font
    size, or to a data source column name referring to a column of font size
    data.

    The ``FontSizeSpec`` property attempts to first interpret string values as
    font sizes (i.e. valid CSS length values). Otherwise string values are
    interpreted as field names. For example:

    .. code-block:: python

        m.font_size = "10pt"  # value

        m.font_size = "1.5em" # value

        m.font_size = "foo"   # field

    A full list of all valid CSS length units can be found here:

    https://drafts.csswg.org/css-values/#lengths

    '''
    _font_size_re = re.compile("^[0-9]+(\.[0-9]+)?(%|em|ex|ch|ic|rem|vw|vh|vi|vb|vmin|vmax|cm|mm|q|in|pc|pt|px)$", re.I)

    def __init__(self, default, help=None):
        super(FontSizeSpec, self).__init__(List(String), default=default, help=help)

    def prepare_value(self, cls, name, value):
        if isinstance(value, string_types) and self._font_size_re.match(value) is not None:
            value = dict(value=value)
        return super(FontSizeSpec, self).prepare_value(cls, name, value)

class UnitsSpec(NumberSpec):
    ''' A base class for numeric :class:`~bokeh.core.properties.DataSpec`
    properties that should also have units.

    '''
    def __init__(self, default, units_type, units_default, help=None):
        super(UnitsSpec, self).__init__(default=default, help=help)
        self._units_type = self._validate_type_param(units_type)
        # this is a hack because we already constructed units_type
        self._units_type.validate(units_default)
        self._units_type._default = units_default
        # this is sort of a hack because we don't have a
        # serialized= kwarg on every Property subtype
        self._units_type._serialized = False

    def make_descriptors(self, base_name):
        units_name = base_name + "_units"
        units_props = self._units_type.make_descriptors(units_name)
        return units_props + [ UnitsSpecPropertyDescriptor(property=self, name=base_name, units_prop=units_props[0]) ]

    def to_serializable(self, obj, name, val):
        d = super(UnitsSpec, self).to_serializable(obj, name, val)
        if d is not None and 'units' not in d:
            # d is a PropertyValueDict at this point, we need to convert it to
            # a plain dict if we are going to modify its value, otherwise a
            # notify_change that should not happen will be triggered
            d = dict(d)
            d["units"] = getattr(obj, name+"_units")
        return d

    def __str__(self):
        return "%s(units_default=%r)" % (self.__class__.__name__, self._units_type._default)

class AngleSpec(UnitsSpec):
    ''' A numeric DataSpec property to represent angles.

    Acceptable values for units are ``"rad"`` and ``"deg"``.

    '''
    def __init__(self, default=None, units_default="rad", help=None):
        super(AngleSpec, self).__init__(default=default, units_type=Enum(enums.AngleUnits), units_default=units_default, help=help)

class DistanceSpec(UnitsSpec):
    ''' A numeric DataSpec property to represent screen or data space distances.

    Acceptable values for units are ``"screen"`` and ``"data"``.

    '''
    def __init__(self, default=None, units_default="data", help=None):
        super(DistanceSpec, self).__init__(default=default, units_type=Enum(enums.SpatialUnits), units_default=units_default, help=help)

    def prepare_value(self, cls, name, value):
        try:
            if value is not None and value < 0:
                raise ValueError("Distances must be positive or None!")
        except TypeError:
            pass
        return super(DistanceSpec, self).prepare_value(cls, name, value)

class ScreenDistanceSpec(NumberSpec):
    ''' A numeric DataSpec property to represent screen distances.

    .. note::
        Units are always ``"screen"``.

    '''
    def to_serializable(self, obj, name, val):
        d = super(ScreenDistanceSpec, self).to_serializable(obj, name, val)
        d["units"] = "screen"
        return d

    def prepare_value(self, cls, name, value):
        try:
            if value is not None and value < 0:
                raise ValueError("Distances must be positive or None!")
        except TypeError:
            pass
        return super(ScreenDistanceSpec, self).prepare_value(cls, name, value)

class DataDistanceSpec(NumberSpec):
    ''' A numeric DataSpec property to represent data space distances.

    .. note::
        Units are always ``"data"``.

    '''
    def to_serializable(self, obj, name, val):
        d = super(ScreenDistanceSpec, self).to_serializable(obj, name, val)
        d["units"] = "data"
        return d

    def prepare_value(self, cls, name, value):
        try:
            if value is not None and value < 0:
                raise ValueError("Distances must be positive or None!")
        except TypeError:
            pass
        return super(DataDistanceSpec, self).prepare_value(cls, name, value)

class ColorSpec(DataSpec):
    ''' A |DataSpec| property that can be set to a fixed value that is a
    |Color|, or a data source column name referring to a column of color
    data.

    The ``ColorSpec`` property attempts to first interpret string values as
    colors. Otherwise, string values are interpreted as field names. For
    example:

    .. code-block:: python

        m.color = "#a4225f"   # value (hex color string)

        m.color = "firebrick" # value (named CSS color string)

        m.color = "foo"       # field (named "foo")

    This automatic interpretation can be override using the dict format
    directly, or by using the |field| function:

    .. code-block:: python

        m.color = { "field": "firebrick" } # field (named "firebrick")

        m.color = field("firebrick")       # field (named "firebrick")

    '''
    def __init__(self, default, help=None):
        super(ColorSpec, self).__init__(Color, default=default, help=help)

    @classmethod
    def isconst(cls, val):
        ''' Whether the value is a string color literal.

        Checks for a well-formed hexadecimal color value or a named color.

        Args:
            val (str) : the value to check

        Returns:
            True, if the value is a string color literal

        '''
        return isinstance(val, string_types) and \
               ((len(val) == 7 and val[0] == "#") or val in enums.NamedColor)

    @classmethod
    def is_color_tuple(cls, val):
        ''' Whether the value is a color tuple.

        Args:
            val (str) : the value to check

        Returns:
            True, if the value is a color tuple

        '''
        return isinstance(val, tuple) and len(val) in (3, 4)

    @classmethod
    def format_tuple(cls, colortuple):
        ''' Convert a color tuple to a CSS RBG(A) value.

        Args:
            colortuple (tuple) : the value to convert

        Returns:
            str: CSS RGB(A) string

        '''
        if len(colortuple) == 3:
            return "rgb%r" % (colortuple,)
        else:
            return "rgba%r" % (colortuple,)

    def to_serializable(self, obj, name, val):
        if val is None:
            return dict(value=None)

        # Check for hexadecimal or named color
        if self.isconst(val):
            return dict(value=val)

        # Check for RGB or RGBa tuple
        if isinstance(val, tuple):
            return dict(value=self.format_tuple(val))

        # Check for data source field name
        if isinstance(val, string_types):
            return dict(field=val)

        # Must be dict, return as-is
        return val

    def validate(self, value):
        try:
            return super(ColorSpec, self).validate(value)
        except ValueError as e:
            # Check for tuple input if not yet a valid input type
            if self.is_color_tuple(value):
                return True
            else:
                raise e

    def transform(self, value):

        # Make sure that any tuple has either three integers, or three integers and one float
        if isinstance(value, tuple):
            value = tuple(int(v) if i < 3 else v for i, v in enumerate(value))

        return value
