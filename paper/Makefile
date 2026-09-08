# Build the draft. This cluster has pdflatex + bibtex but NOT latexmk, so the
# passes are spelled out. Two pdflatex passes resolve \ref/\label; a third is
# only needed once refs.bib is wired up (see main.tex).
#
#   make          build main.pdf
#   make results  regenerate tables + figures from the recovered data, then build
#   make clean    remove build artifacts (keeps main.pdf)
#   make distclean also remove main.pdf

PY   := /u/ad11/miniconda3/envs/prompt_opt/bin/python
TEX  := pdflatex -interaction=nonstopmode -halt-on-error
SRC  := main.tex $(wildcard sections/*.tex) $(wildcard tables/*.tex) $(wildcard figures/*.pdf)

.PHONY: all results clean distclean

all: main.pdf

main.pdf: $(SRC)
	$(TEX) main.tex
	$(TEX) main.tex
	@echo
	@echo "built main.pdf"
	@grep -c 'TODO' main.log >/dev/null 2>&1 || true
	@echo "open TODOs in the draft:"
	@grep -o 'TODO:[^]]*' $(wildcard sections/*.tex) main.tex | head -20 || true

# Regenerate every number and figure from results/v3_wandb_export.json. Run
# analysis/recover_from_wandb.py first if the export itself is stale.
results:
	cd .. && $(PY) analysis/paper_results.py --tex
	$(MAKE) all

clean:
	rm -f main.aux main.log main.out main.toc main.bbl main.blg \
	      sections/*.aux

distclean: clean
	rm -f main.pdf
