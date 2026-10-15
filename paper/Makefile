# Makefile for compiling RetinaGuard IEEE LaTeX Paper

TARGET = main
PDFLATEX = pdflatex
BIBTEX = bibtex

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex refs.bib
	$(PDFLATEX) $(TARGET).tex
	$(BIBTEX) $(TARGET)
	$(PDFLATEX) $(TARGET).tex
	$(PDFLATEX) $(TARGET).tex

clean:
	rm -f $(TARGET).aux $(TARGET).bbl $(TARGET).blg $(TARGET).log $(TARGET).out $(TARGET).pdf

.PHONY: all clean
