# The GUI submodule is intentionally not re-exported here: importing the package
# (or impasse.position / impasse.ai) should not pull in pygame. Import impasse.gui
# explicitly when the graphical interface is needed.
from impasse.ai import *
from impasse.position import *
