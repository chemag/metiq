VENV_DIR ?= .venv
VENV_BASE_PYTHON ?= python3
VENV_CREATE_ARGS ?=
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_STAMP := $(VENV_DIR)/.requirements-installed

.PHONY: venv venv-verify

venv: $(VENV_STAMP)

$(VENV_PYTHON):
	$(VENV_BASE_PYTHON) -m venv $(VENV_CREATE_ARGS) $(VENV_DIR)

$(VENV_STAMP): requirements.txt | $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install -r requirements.txt
	touch $@

venv-verify: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip check
	PYTHONPATH=src $(VENV_PYTHON) -c 'import cv2; import metiq; print("OK: metiq and its dependencies import successfully")'


VALUE=22

all: \
    results/linux.mp4.csv \
    results/mbp.mp4.csv \
    results/pixel5.mp4.csv \
    results/pixel6a.mp4.csv \
    results/pixel6a.bt.mp4.csv \
    results/pixel6a.bt.1m.mp4.csv \
    results/pixel6a.bt.wall.mp4.csv \
    results/tate.mp4.csv \
    results/sn.evt1.mp4.csv \
    results/stella.mp4.csv \
    parse.5x4 \
    parse.7x5 \
    parse.9x6 \
    parse.9x8 \
    results/metiq.mp4.csv \
    results/metiq.20fps.mp4.csv \
    results/metiq.60fps.mp4.csv


VERSION=$(shell ./src/_version.py)

tar: metiq.${VERSION}.tar.gz

.PHONY: test
test:
	test/tests.py


metiq.${VERSION}.tar.gz:
	tar cvf metiq.${VERSION}.tar Makefile README.md ./src/*py
	gzip -f metiq.${VERSION}.tar


doc.zip: README.md
	zip -r $@ README.md doc/


README.html: README.md
	pandoc README.md -o README.html

README.pdf: README.md
	pandoc README.md -o README.pdf

results/metiq.mp4.csv: results/metiq.mp4
	./src/metiq.py parse -i $^ -o $@

results/metiq.mp4:
	./src/metiq.py generate -o $@

results/metiq.20fps.mp4: results/metiq.mp4
	ffmpeg -i $^ -filter:v minterpolate=fps=20 $@

results/metiq.60fps.mp4: results/metiq.mp4
	ffmpeg -i $^ -filter:v minterpolate=fps=60 $@

results/metiq.20fps.mp4.csv: results/metiq.20fps.mp4
	./src/metiq.py parse -i $^ -o $@

results/metiq.60fps.mp4.csv: results/metiq.60fps.mp4
	./src/metiq.py parse -i $^ -o $@

results/linux.mp4.csv: results/linux.mp4
	./src/metiq.py parse -i $^ -o $@ --luma-threshold 20

results/mbp.mp4.csv: results/mbp.mp4
	./src/metiq.py parse -i $^ -o $@ --luma-threshold 20

results/pixel5.mp4.csv: results/pixel5.mp4
	./src/metiq.py parse -i $^ -o $@ --luma-threshold 20

results/pixel6a.mp4.csv: results/pixel6a.mp4
	./src/metiq.py parse -i $^ -o $@ --luma-threshold 20

results/pixel6a.bt.mp4.csv: results/pixel6a.bt.mp4
	./src/metiq.py parse -i $^ -o $@ --luma-threshold 20

results/pixel6a.bt.1m.mp4.csv: results/pixel6a.bt.1m.mp4
	./src/metiq.py parse -i $^ -o $@ --luma-threshold 20

results/pixel6a.bt.wall.mp4.csv: results/pixel6a.bt.wall.mp4
	./src/metiq.py parse -i $^ -o $@ --luma-threshold 20

results/tate.mp4.csv: results/tate.mp4
	./src/metiq.py parse -i $^ -o $@ --luma-threshold 20

results/sn.evt1.mp4.csv: results/sn.evt1.mp4
	./src/metiq.py parse -i $^ -o $@ --luma-threshold 20

results/stella.mp4.csv: results/stella.mp4
	./src/metiq.py parse -i $^ -o $@ --luma-threshold 20


parse.5x4: doc/vft.5x4.${VALUE}.png
	./src/vft.py parse -i $^

parse.7x5: doc/vft.7x5.${VALUE}.png
	./src/vft.py parse -i $^

parse.9x6: doc/vft.9x6.${VALUE}.png
	./src/vft.py parse -i $^

parse.9x8: doc/vft.9x8.${VALUE}.png
	./src/vft.py parse -i $^


doc/vft.5x4.${VALUE}.png:
	./src/vft.py generate -o $@ --vft-id 5x4 --value ${VALUE}

doc/vft.7x5.${VALUE}.png:
	./src/vft.py generate -o $@ --vft-id 7x5 --value ${VALUE}

doc/vft.9x6.${VALUE}.png:
	./src/vft.py generate -o $@ --vft-id 9x6 --value ${VALUE}

doc/vft.9x8.${VALUE}.png:
	./src/vft.py generate -o $@ --vft-id 9x8 --value ${VALUE}


NUMBERS = 0 1 2 3 4 5 6
write_all:
	$(foreach var,$(NUMBERS),./src/vft.py generate -o /tmp/vft.5x4.$(var).png --vft-id 5x4 --value $(var);)

read_all:
	$(foreach var,$(NUMBERS),./src/vft.py parse -i /tmp/vft.5x4.$(var).png;)



clean:
	\rm -rf \
    doc/vft.5x4.${VALUE}.png \
    doc/vft.7x5.${VALUE}.png \
    doc/vft.9x6.${VALUE}.png \
    doc/vft.9x8.${VALUE}.png
