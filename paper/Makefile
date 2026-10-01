# Build the draft. This cluster has pdflatex + bibtex but NOT latexmk, so the
# passes are spelled out: pdflatex, bibtex, then two more pdflatex passes to
# resolve citations and \ref/\label.
#
#   make          build main.pdf
#   make results  regenerate tables + figures from the recovered data, then build
#   make clean    remove build artifacts (keeps main.pdf)
#   make distclean also remove main.pdf

PY   := /u/ad11/miniconda3/envs/prompt_opt/bin/python
PY3  := /u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
TEX  := pdflatex -interaction=nonstopmode -halt-on-error
SRC  := main.tex refs.bib $(wildcard sections/*.tex) $(wildcard tables/*.tex) $(wildcard figures/*.pdf)

.PHONY: all results clean distclean

all: main.pdf

main.pdf: $(SRC)
	$(TEX) main.tex
	bibtex main
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
	cd .. && $(PY3) analysis/full_eval.py --campaign v4
	cd .. && $(PY3) analysis/full_eval.py --campaign v5
	cd .. && $(PY3) analysis/reward_scale_analysis.py
	cd .. && $(PY3) analysis/theory_checks.py --fig paper/figures/theory_toy.pdf
	# Reward I diagnosis (needs results/v4/frontier_probe, from slurm/frontier_probe.slurm)
	cd .. && $(PY3) analysis/frontier_diagnosis.py
	cd .. && $(PY3) analysis/reward_design.py
	cd .. && $(PY3) analysis/diagnosis_macros.py
	cd .. && $(PY3) analysis/constraint_sim.py
	# worked example (needs results/examples, from slurm/example_outputs.slurm)
	cd .. && $(PY3) analysis/example_table.py
	$(MAKE) all

clean:
	rm -f main.aux main.log main.out main.toc main.bbl main.blg \
	      sections/*.aux

distclean: clean
	rm -f main.pdf
