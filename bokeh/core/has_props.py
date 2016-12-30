'''

.. note::
    These classes form part of the very low-level machinery that implements
    the Bokeh model and property system. It is unlikely that any of these
    classes or their methods will be applicable to any standard usage or to
    anyone who is not directly developing on Bokeh's own infrastructure.

'''
from __future__ import absolute_import

import logging
logger = logging.getLogger(__name__)

import difflib
import inspect
from operator import itemgetter
import sys
from warnings import warn

from six import StringIO

from ..util.dependencies import import_optional
from ..util.future import with_metaclass
from ..util.string import nice_join
from .property.containers import PropertyValueContainer
from .property.factory import PropertyFactory
from .property.override import Override

IPython = import_optional('IPython')

if IPython:
    from IPython.lib.pretty import RepresentationPrinter

    class BokehPrettyPrinter(RepresentationPrinter):
        def __init__(self, output, verbose=False, max_width=79, newline='\n'):
            super(BokehPrettyPrinter, self).__init__(output, verbose, max_width, newline)
            self.type_pprinters[HasProps] = lambda obj, p, cycle: obj._bokeh_repr_pretty_(p, cycle)

_EXAMPLE_TEMPLATE = '''

    Example
    -------

    .. bokeh-plot:: ../%(path)s
        :source-position: none

    *source:* `%(path)s <https://github.com/bokeh/bokeh/tree/master/%(path)s>`_

'''

class MetaHasProps(type):
    def __new__(meta_cls, class_name, bases, class_dict):
        names_with_refs = set()
        container_names = set()

        # Now handle all the Override
        overridden_defaults = {}
        for name, prop in class_dict.items():
            if not isinstance(prop, Override):
                continue
            if prop.default_overridden:
                overridden_defaults[name] = prop.default

        for name, default in overridden_defaults.items():
            del class_dict[name]

        generators = dict()
        for name, generator in class_dict.items():
            if isinstance(generator, PropertyFactory):
                generators[name] = generator
            elif isinstance(generator, type) and issubclass(generator, PropertyFactory):
                # Support the user adding a property without using parens,
                # i.e. using just the Property subclass instead of an
                # instance of the subclass
                generators[name] = generator.autocreate()

        dataspecs = {}
        new_class_attrs = {}

        for name, generator in generators.items():
            prop_descriptors = generator.make_descriptors(name)
            replaced_self = False
            for prop_descriptor in prop_descriptors:
                if prop_descriptor.name in generators:
                    if generators[prop_descriptor.name] is generator:
                        # a generator can replace itself, this is the
                        # standard case like `foo = Int()`
                        replaced_self = True
                        prop_descriptor.add_prop_descriptor_to_class(class_name, new_class_attrs, names_with_refs, container_names, dataspecs)
                    else:
                        # if a generator tries to overwrite another
                        # generator that's been explicitly provided,
                        # use the prop that was manually provided
                        # and ignore this one.
                        pass
                else:
                    prop_descriptor.add_prop_descriptor_to_class(class_name, new_class_attrs, names_with_refs, container_names, dataspecs)
            # if we won't overwrite ourselves anyway, delete the generator
            if not replaced_self:
                del class_dict[name]

        class_dict.update(new_class_attrs)

        class_dict["__properties__"] = set(new_class_attrs)
        class_dict["__properties_with_refs__"] = names_with_refs
        class_dict["__container_props__"] = container_names
        if len(overridden_defaults) > 0:
            class_dict["__overridden_defaults__"] = overridden_defaults
        if dataspecs:
            class_dict["__dataspecs__"] = dataspecs

        if "__example__" in class_dict:
            path = class_dict["__example__"]
            class_dict["__doc__"] += _EXAMPLE_TEMPLATE % dict(path=path)

        return super(MetaHasProps, meta_cls).__new__(meta_cls, class_name, bases, class_dict)

    def __init__(cls, class_name, bases, nmspc):
        if class_name == 'HasProps':
            return
        # Check for improperly overriding a Property attribute.
        # Overriding makes no sense except through the Override
        # class which can be used to tweak the default.
        # Historically code also tried changing the Property's
        # type or changing from Property to non-Property: these
        # overrides are bad conceptually because the type of a
        # read-write property is invariant.
        cls_attrs = cls.__dict__.keys() # we do NOT want inherited attrs here
        for attr in cls_attrs:
            for base in bases:
                if issubclass(base, HasProps) and attr in base.properties():
                    warn(('Property "%s" in class %s was overridden by a class attribute ' + \
                          '"%s" in class %s; it never makes sense to do this. ' + \
                          'Either %s.%s or %s.%s should be removed, or %s.%s should not ' + \
                          'be a Property, or use Override(), depending on the intended effect.') %
                         (attr, base.__name__, attr, class_name,
                          base.__name__, attr,
                          class_name, attr,
                          base.__name__, attr),
                         RuntimeWarning, stacklevel=2)

        if "__overridden_defaults__" in cls.__dict__:
            our_props = cls.properties()
            for key in cls.__dict__["__overridden_defaults__"].keys():
                if key not in our_props:
                    warn(('Override() of %s in class %s does not override anything.') % (key, class_name),
                         RuntimeWarning, stacklevel=2)

def accumulate_from_superclasses(cls, propname):
    cachename = "__cached_all" + propname
    # we MUST use cls.__dict__ NOT hasattr(). hasattr() would also look at base
    # classes, and the cache must be separate for each class
    if cachename not in cls.__dict__:
        s = set()
        for c in inspect.getmro(cls):
            if issubclass(c, HasProps) and hasattr(c, propname):
                base = getattr(c, propname)
                s.update(base)
        setattr(cls, cachename, s)
    return cls.__dict__[cachename]

def accumulate_dict_from_superclasses(cls, propname):
    cachename = "__cached_all" + propname
    # we MUST use cls.__dict__ NOT hasattr(). hasattr() would also look at base
    # classes, and the cache must be separate for each class
    if cachename not in cls.__dict__:
        d = dict()
        for c in inspect.getmro(cls):
            if issubclass(c, HasProps) and hasattr(c, propname):
                base = getattr(c, propname)
                for k,v in base.items():
                    if k not in d:
                        d[k] = v
        setattr(cls, cachename, d)
    return cls.__dict__[cachename]

class HasProps(with_metaclass(MetaHasProps, object)):
    ''' Base class for all class types that have Bokeh properties.

    '''
    def __init__(self, **properties):
        super(HasProps, self).__init__()
        self._property_values = dict()

        for name, value in properties.items():
            setattr(self, name, value)

    def equals(self, other):
        ''' Structural equality of models. '''
        # NOTE: don't try to use this to implement __eq__. Because then
        # you will be tempted to implement __hash__, which would interfere
        # with mutability of models. However, not implementing __hash__
        # will make bokeh unusable in Python 3, where proper implementation
        # of __hash__ is required when implementing __eq__.
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.properties_with_values() == other.properties_with_values()

    def __setattr__(self, name, value):
        # self.properties() below can be expensive so avoid it
        # if we're just setting a private underscore field
        if name.startswith("_"):
            super(HasProps, self).__setattr__(name, value)
            return

        props = sorted(self.properties())
        deprecated = getattr(self, '__deprecated_attributes__', [])

        if name in props or name in deprecated:
            super(HasProps, self).__setattr__(name, value)
        else:
            matches, text = difflib.get_close_matches(name.lower(), props), "similar"

            if not matches:
                matches, text = props, "possible"

            raise AttributeError("unexpected attribute '%s' to %s, %s attributes are %s" %
                (name, self.__class__.__name__, text, nice_join(matches)))

    def set_from_json(self, name, json, models=None, setter=None):
        ''' Sets a property of the object using JSON and a dictionary mapping
        model ids to model instances. The model instances are necessary if the
        JSON contains references to models.

        '''
        if name in self.properties():
            #logger.debug("Patching attribute %s of %r", attr, patched_obj)
            prop = self.lookup(name)
            prop.set_from_json(self, json, models, setter)
        else:
            logger.warn("JSON had attr %r on obj %r, which is a client-only or invalid attribute that shouldn't have been sent", name, self)

    def update(self, **kwargs):
        ''' Updates the object's properties from the given keyword args. '''
        for k,v in kwargs.items():
            setattr(self, k, v)

    def update_from_json(self, json_attributes, models=None, setter=None):
        ''' Updates the object's properties from a JSON attributes dictionary. '''
        for k, v in json_attributes.items():
            self.set_from_json(k, v, models, setter)

    def _clone(self):
        ''' Returns a duplicate of this object with all its properties
        set appropriately.  Values which are containers are shallow-copied.
        '''
        return self.__class__(**self._property_values)

    @classmethod
    def lookup(cls, name):
        return getattr(cls, name)

    @classmethod
    def properties_with_refs(cls):
        ''' Return a set of the names of this object's properties that
        have references. We traverse the class hierarchy and
        pull together the full list of properties.
        '''
        return accumulate_from_superclasses(cls, "__properties_with_refs__")

    @classmethod
    def properties_containers(cls):
        ''' Returns a list of properties that are containers.
        '''
        return accumulate_from_superclasses(cls, "__container_props__")

    @classmethod
    def properties(cls, with_bases=True):
        '''Return a set of the names of this object's properties. If
        ``with_bases`` is True, we traverse the class hierarchy
        and pull together the full list of properties; if False,
        we only return the properties introduced in the class
        itself.

        Args:
           with_bases (bool, optional) :
            Whether to include properties that haven't been set. (default: True)

        Returns:
           a set of property names

        '''
        if with_bases:
            return accumulate_from_superclasses(cls, "__properties__")
        else:
            return set(cls.__properties__)

    @classmethod
    def _overridden_defaults(cls):
        ''' Returns a dictionary of defaults that have been overridden; this is an implementation detail of Property. '''
        return accumulate_dict_from_superclasses(cls, "__overridden_defaults__")

    @classmethod
    def dataspecs(cls):
        ''' Returns a set of the names of this object's dataspecs (and
        dataspec subclasses).  Traverses the class hierarchy.
        '''
        return set(cls.dataspecs_with_props().keys())

    @classmethod
    def dataspecs_with_props(cls):
        ''' Returns a dict of dataspec names to dataspec properties. '''
        return accumulate_dict_from_superclasses(cls, "__dataspecs__")

    def properties_with_values(self, include_defaults=True):
        ''' Return a dict from property names to the current values of those
        properties.

        Non-serializable properties are skipped and property values are in
        "serialized" format which may be slightly different from the values
        you would normally read from the properties; the intent of this method
        is to return the information needed to losslessly reconstitute the
        object instance.

        Args:
            include_defaults (bool, optional) :
                Whether to include properties that haven't been set. (default: True)

        Returns:
           dict : mapping from property names to their values

        '''
        return self.query_properties_with_values(lambda prop: prop.serialized, include_defaults)

    def query_properties_with_values(self, query, include_defaults=True):
        result = dict()
        if include_defaults:
            keys = self.properties()
        else:
            keys = set(self._property_values.keys())
            if self.themed_values():
                keys |= set(self.themed_values().keys())

        for key in keys:
            prop = self.lookup(key)
            if not query(prop):
                continue

            value = prop.serializable_value(self)
            if not include_defaults:
                if isinstance(value, PropertyValueContainer) and value._unmodified_default_value:
                    continue
            result[key] = value

        return result

    def set(self, **kwargs):
        ''' Sets a number of properties at once '''
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def themed_values(self):
        ''' Get any theme-provided overrides as a dict from property name
        to value, or None if no theme overrides any values for this instance.

        '''
        if hasattr(self, '__themed_values__'):
            return getattr(self, '__themed_values__')
        else:
            return None

    def apply_theme(self, property_values):
        ''' Apply a set of theme values which will be used rather than
        defaults, but will not override application-set values.

        The passed-in dictionary may be kept around as-is and shared with
        other instances to save memory (so neither the caller nor the
        |HasProps| instance should modify it).

        .. |HasProps| replace:: :class:`~bokeh.properties.HasProps`

        '''
        old_dict = None
        if hasattr(self, '__themed_values__'):
            old_dict = getattr(self, '__themed_values__')

        # if the same theme is set again, it should reuse the
        # same dict
        if old_dict is property_values:
            return

        removed = set()
        # we're doing a little song-and-dance to avoid storing __themed_values__ or
        # an empty dict, if there's no theme that applies to this HasProps instance.
        if old_dict is not None:
            removed.update(set(old_dict.keys()))
        added = set(property_values.keys())
        old_values = dict()
        for k in added.union(removed):
            old_values[k] = getattr(self, k)

        if len(property_values) > 0:
            setattr(self, '__themed_values__', property_values)
        elif hasattr(self, '__themed_values__'):
            delattr(self, '__themed_values__')

        # Emit any change notifications that result
        for k, v in old_values.items():
            prop = self.lookup(k)
            prop.trigger_if_changed(self, v)

    def unapply_theme(self):
        self.apply_theme(property_values=dict())

    def __str__(self):
        return "%s(...)" % self.__class__.__name__

    __repr__ = __str__

    def _bokeh_repr_pretty_(self, p, cycle):
        name = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)

        if cycle:
            p.text("%s(...)" % name)
        else:
            with p.group(4, '%s(' % name, ')'):
                props = self.properties_with_values().items()
                sorted_props = sorted(props, key=itemgetter(0))
                all_props = sorted_props
                for i, (prop, value) in enumerate(all_props):
                    if i == 0:
                        p.breakable('')
                    else:
                        p.text(',')
                        p.breakable()
                    p.text(prop)
                    p.text('=')
                    p.pretty(value)

    def pretty(self, verbose=False, max_width=79, newline='\n'):
        ''' Pretty print the object's representation. '''
        if not IPython:
            cls = self.__class.__
            raise RuntimeError("%s.%s.pretty() requires IPython" % (cls.__module__, cls.__name__))
        else:
            stream = StringIO()
            printer = BokehPrettyPrinter(stream, verbose, max_width, newline)
            printer.pretty(self)
            printer.flush()
            return stream.getvalue()

    def pprint(self, verbose=False, max_width=79, newline='\n'):
        ''' Like `pretty` but print to stdout. '''
        if not IPython:
            cls = self.__class.__
            raise RuntimeError("%s.%s.pretty() requires IPython" % (cls.__module__, cls.__name__))
        else:
            printer = BokehPrettyPrinter(sys.stdout, verbose, max_width, newline)
            printer.pretty(self)
            printer.flush()
            sys.stdout.write(newline)
            sys.stdout.flush()
