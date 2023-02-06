
VALUE=22

all: \
    analyze.5x4 \
    analyze.7x5 \
    analyze.9x6 \
    analyze.9x8 \
    results/metiq.mp4.csv \
    results/metiq.20fps.mp4.csv \
    results/metiq.60fps.mp4.csv


VERSION=$(shell ./_version.py)

metiq.${VERSION}.tar.gz:
	tar cvf metiq.${VERSION}.tar Makefile README.md aruco_common.py audio_analyze.py audio_common.py audio_generate.py common.py metiq.py _version.py vft.py video_analyze.py video_common.py video_generate.py
	gzip -f metiq.${VERSION}.tar


README.html: README.md
	pandoc README.md -o README.html

README.pdf: README.md
	pandoc README.md -o README.pdf

results/metiq.mp4.csv: results/metiq.mp4
	./metiq.py analyze -i $^ -o $@

results/metiq.mp4:
	./metiq.py generate -o $@

results/metiq.20fps.mp4: results/metiq.mp4
	ffmpeg -i $^ -filter:v minterpolate=fps=20 $@

results/metiq.60fps.mp4: results/metiq.mp4
	ffmpeg -i $^ -filter:v minterpolate=fps=60 $@

results/metiq.20fps.mp4.csv: results/metiq.20fps.mp4
	./metiq.py analyze -i $^ -o $@

results/metiq.60fps.mp4.csv: results/metiq.60fps.mp4
	./metiq.py analyze -i $^ -o $@


analyze.5x4: doc/vft.5x4.${VALUE}.png
	./vft.py analyze -i $^

analyze.7x5: doc/vft.7x5.${VALUE}.png
	./vft.py analyze -i $^

analyze.9x6: doc/vft.9x6.${VALUE}.png
	./vft.py analyze -i $^

analyze.9x8: doc/vft.9x8.${VALUE}.png
	./vft.py analyze -i $^


doc/vft.5x4.${VALUE}.png:
	./vft.py generate -o $@ --vft-id 5x4 --value ${VALUE}

doc/vft.7x5.${VALUE}.png:
	./vft.py generate -o $@ --vft-id 7x5 --value ${VALUE}

doc/vft.9x6.${VALUE}.png:
	./vft.py generate -o $@ --vft-id 9x6 --value ${VALUE}

doc/vft.9x8.${VALUE}.png:
	./vft.py generate -o $@ --vft-id 9x8 --value ${VALUE}


NUMBERS = 0 1 2 3 4 5 6
write_all:
	$(foreach var,$(NUMBERS),./vft.py generate -o /tmp/vft.5x4.$(var).png --vft-id 5x4 --value $(var);)

read_all:
	$(foreach var,$(NUMBERS),./vft.py analyze -i /tmp/vft.5x4.$(var).png;)



clean:
	\rm -rf \
    doc/vft.5x4.${VALUE}.png \
    doc/vft.7x5.${VALUE}.png \
    doc/vft.9x6.${VALUE}.png \
    doc/vft.9x8.${VALUE}.png
