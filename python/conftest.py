# Root conftest: anchors pytest's rootdir at python/ so that running
#   pytest tests/test_detection.py
# from this directory puts python/ on sys.path and `import celldetect`
# resolves to the student package. No fixtures needed.
