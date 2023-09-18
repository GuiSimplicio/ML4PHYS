import pytest
import sys 
sys.path.append("assignment-0-GuiSimplicio")
import main as main

def test_hello():
    assert main.hello_world() == "Hello World!"

@pytest.mark.parametrize("x,y,expected", [(1,2,3), (2,2,4), (3,-2,1)])
def test_add(x,y,expected):
    assert main.add(x,y) == expected
