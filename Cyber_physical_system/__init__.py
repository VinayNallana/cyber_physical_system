# Shim package to make 'cyber_physical_system' import point to the
# existing 'Cyber_physical_system' folder when present on disk.
import os

_this_dir = os.path.dirname(__file__)
_alt = os.path.abspath(os.path.join(_this_dir, "..", "Cyber_physical_system"))
if os.path.isdir(_alt):
    __path__.insert(0, _alt)
