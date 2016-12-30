from mock import patch
import pytest

import bokeh.core.property.descriptors as pd

def test_PropertyDescriptor__init__():
    d = pd.PropertyDescriptor("foo")
    assert d.name == "foo"

def test_PropertyDescriptor__str__():
    d = pd.PropertyDescriptor("foo")
    assert str(d) == "PropertyDescriptor(foo)"

def test_PropertyDescriptor_abstract():
    d = pd.PropertyDescriptor("foo")
    class Foo(object): pass
    f = Foo()
    with pytest.raises(NotImplementedError):
        d.__get__(f)

    with pytest.raises(NotImplementedError):
        d.__set__(f, 11)

    with pytest.raises(NotImplementedError):
        d.__delete__(f)

    with pytest.raises(NotImplementedError):
        d.class_default(f)

    with pytest.raises(NotImplementedError):
        d.serialized

    with pytest.raises(NotImplementedError):
        d.readonly

    with pytest.raises(NotImplementedError):
        d.has_ref

    with pytest.raises(NotImplementedError):
        d.trigger_if_changed(f, 11)

    with pytest.raises(NotImplementedError):
        d._internal_set(f, 11)

@patch('bokeh.core.property.descriptors.PropertyDescriptor._internal_set')
def test_PropertyDescriptor_set_from_json(mock_iset):
    class Foo(object): pass
    f = Foo()
    d = pd.PropertyDescriptor("foo")
    d.set_from_json(f, "bar", 10)
    assert mock_iset.called_once_with((f, "bar", 10), {})

def test_PropertyDescriptor_serializable_value():
    result = {}
    class Foo(object):
        def serialize_value(self, val):
            result['foo'] = val
    f = Foo()
    f.foo = 10

    d = pd.PropertyDescriptor("foo")
    d.property = Foo()

    # simulate the __get__ a subclass would have
    d.__get__ = lambda self: f.foo

    d.serializable_value(f)
    assert result['foo'] == 10

def test_add_prop_descriptor_to_class():
    pass




def test_BasicPropertyDescriptor__init__():
    class Foo(object):
        '''doc'''
        pass
    f = Foo()
    d = pd.BasicPropertyDescriptor(f, "foo")
    assert d.name == "foo"
    assert d.property == f
    assert d.__doc__ == f.__doc__

def test_BasicPropertyDescriptor__str__():
    class Foo(object): pass
    f = Foo()
    d = pd.BasicPropertyDescriptor(f, "foo")
    assert str(d) == str(f)

def test_BasicPropertyDescriptor_serialized():
    class Foo(object): pass
    f = Foo()
    f.serialized = "stuff"
    d = pd.BasicPropertyDescriptor(f, "foo")
    assert d.serialized == "stuff"

def test_BasicPropertyDescriptor_readonly():
    class Foo(object): pass
    f = Foo()
    f.readonly = "stuff"
    d = pd.BasicPropertyDescriptor(f, "foo")
    assert d.readonly == "stuff"

def test_BasicPropertyDescriptor_has_ref():
    class Foo(object): pass
    f = Foo()
    f.has_ref = "stuff"
    d = pd.BasicPropertyDescriptor(f, "foo")
    assert d.has_ref == "stuff"




def test_UnitsSpecPropertyDescriptor__init__():
    class Foo(object):
        '''doc'''
        pass
    f = Foo()
    g = Foo()
    d = pd.UnitsSpecPropertyDescriptor(f, "foo", g)
    assert d.name == "foo"
    assert d.property == f
    assert d.__doc__ == f.__doc__
    assert d.units_prop == g
