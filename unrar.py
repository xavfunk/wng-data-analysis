#import patoolib
import os
from pyunpack import Archive
#patoolib.extract_archive("Pilot6.rar", outdir = os.getcwd())
Archive("Pilot6.rar").extractall(os.getcwd())
