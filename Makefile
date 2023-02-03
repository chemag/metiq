
VALUE=22

all: \
    analyze.5x4 \
    analyze.7x5 \
    analyze.9x6 \
    analyze.9x8


test:
	./metiq.py generate -o /tmp/foo.mp4
	./metiq.py analyze -i /tmp/foo.mp4 -o /tmp/foo.csv


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
