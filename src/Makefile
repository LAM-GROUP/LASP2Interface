# LASP2 installation makefile

SHELL=/bin/bash
PYTHON=python3

PROJECT_DIR=$(abspath ..)
PROJECT_BIN=$(PROJECT_DIR)/bin
PROJECT_LIB=$(PROJECT_DIR)/lib
PROJECT_ENV=$(PROJECT_LIB)/env



# Installation directory for binaries.
INSTALL_BIN=$(HOME)/local/bin

##########
# Installation rule #
##########
install:
	@$(PYTHON) -m venv $(PROJECT_ENV)
	. $(PROJECT_ENV)/bin/activate; pip install --upgrade pip; pip install --no-cache-dir -Ur ../lib/requirements.txt
	. $(PROJECT_ENV)/bin/activate; pyinstaller interfaceLAMMPS.py --distpath ../bin --noconfirm
	sed -i "s|^dirInterface = '[^']*'|dirInterface = '$(PROJECT_BIN)/interfaceLAMMPS/interfaceLAMMPS'|g" LASP2.py
	. $(PROJECT_ENV)/bin/activate; pyinstaller LASP2.py --distpath ../bin --noconfirm
	cd ../bin; ln -s LASP2/LASP2 ./lasp2