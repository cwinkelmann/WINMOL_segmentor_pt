def test_package_imports():
    import winmol_unet
    assert hasattr(winmol_unet, "__version__")
