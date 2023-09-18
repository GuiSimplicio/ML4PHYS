from re import X
import pytest
import numpy as np
import sys 
sys.path.append("assignment-0-GuiSimplicio")
import main as main

def test_torch():
    assert main.torch_version() == "1.12.1"

def test_numpy():
    assert main.to_numpy([1,2,3]).shape == (3,)
    assert main.to_numpy([1,2,3]).dtype == "int64"
    assert main.to_numpy([[1,2,3],[4,5,6]]).shape == (2,3)


def test_regression():
    coeff = main.simple_regression()
    assert coeff.shape == (2,)
    assert np.allclose(coeff, [0.5, 0.5])

def test_plotting():
    # make sure that the image gets plotted
    main.plot_gaussian_data()
