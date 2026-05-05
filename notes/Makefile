TEXFILE = main
PDFLATEX = pdflatex

TEXSRC = $(TEXFILE).tex
PDF = $(TEXFILE).pdf

# Default target
all: pdf

# Rule to generate the PDF
pdf: $(TEXSRC)
	$(PDFLATEX) $(TEXSRC)

clean:
	rm -f $(TEXFILE).aux $(TEXFILE).bbl $(TEXFILE).bcf $(TEXFILE).blg \
	$(TEXFILE).log $(TEXFILE).out $(TEXFILE).run.xml $(TEXFILE).toc

cleanall: clean
	rm -f $(PDF)

.PHONY: all clean cleanall
