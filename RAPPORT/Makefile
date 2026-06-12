MAIN = rapport

.PHONY: all clean

all:
	latexmk -pdf -interaction=nonstopmode -outdir=build $(MAIN).tex
	cp build/$(MAIN).pdf .

clean:
	latexmk -C -outdir=build $(MAIN).tex
	rm -f $(MAIN).pdf
