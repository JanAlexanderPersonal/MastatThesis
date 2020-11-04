#!/bin/bash
# run the latest pdf container (interactive)

bibtex thesis
latex thesis.tex
pdflatex thesis.tex

rm *.aux
rm *.out
rm *.toc
rm *.log
rm *.fls