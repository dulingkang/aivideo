
import sys
import os

try:
    import flux
    print(f"Flux imported from: {os.path.dirname(flux.__file__)}")
    import flux.util
    print(f"Flux util imported from: {flux.util.__file__}")
except ImportError as e:
    print(f"Could not import flux: {e}")
except Exception as e:
    print(f"Error: {e}")

print("Sys path:", sys.path)
